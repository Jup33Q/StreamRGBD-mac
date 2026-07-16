#!/usr/bin/env python3
"""
StreamDiffusion for Mac — Inference Worker

A headless, camera-agnostic Python backend that reads raw RGB frames from
stdin, runs the CoreML img2img pipeline, and writes JPEG-encoded results to
stdout.

It is designed to be spawned as a CLI command from the Elixir side via an
Erlang Port. The wire protocol is length-prefixed (Erlang {packet, 4}):

  Input frame packet: <<width::32-little, height::32-little, rgb::binary>>
  Input prompt packet: <<0xFFFFFFFF::32-little, prompt_len::32-little, prompt::binary>>
  Output packet:       <<jpeg_size::32-little, jpeg::binary>>

Usage:
    python inference_worker.py --prompt "oil painting style, masterpiece"
"""
import os
import sys
import signal
import struct
import argparse
import threading

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipelines.coreml import Pipeline, COREML_DIR
from configs import MODEL_CONFIGS


# Width sentinel used to signal a prompt update packet instead of a frame.
_PROMPT_UPDATE_SENTINEL = 0xFFFFFFFF
# Depth frame sentinel used to signal a depth frame packet on stdout.
_DEPTH_FRAME_SENTINEL = 0xFFFFFFFE

# -----------------------------------------------------------------------------
# Wire protocol helpers
# -----------------------------------------------------------------------------

def read_exact(n):
    """Read exactly n bytes from stdin, returning None on EOF."""
    data = b""
    while len(data) < n:
        chunk = sys.stdin.buffer.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def read_packet():
    """Read one packet from stdin.

    Returns one of:
      - ("frame", np.ndarray)   – a decoded RGB frame
      - ("prompt", str)         – a runtime prompt update
      - None                    – EOF
    """
    header = read_exact(8)
    if header is None:
        return None
    width, second = struct.unpack("<II", header)

    # Prompt update command: width == 0xFFFFFFFF means this is not a frame.
    if width == _PROMPT_UPDATE_SENTINEL:
        prompt_len = second
        prompt_bytes = read_exact(prompt_len)
        if prompt_bytes is None:
            return None
        return ("prompt", prompt_bytes.decode("utf-8"))

    # Normal RGB frame packet.
    height = second
    pixel_count = width * height * 3
    pixels = read_exact(pixel_count)
    if pixels is None:
        return None
    return ("frame", np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3))


def write_frame(frame_rgb, quality=85):
    """Encode an RGB frame to JPEG and write it length-prefixed to stdout."""
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return False
    jpeg = encoded.tobytes()
    sys.stdout.buffer.write(struct.pack("<I", len(jpeg)))
    sys.stdout.buffer.write(jpeg)
    sys.stdout.buffer.flush()
    return True


def write_depth_frame(depth_u8, quality=85):
    """Encode a grayscale depth frame to JPEG and write it length-prefixed to stdout.

    The packet uses 0xFFFFFFFE as the first length field to signal that this is
    a depth frame rather than a regular RGB frame.
    """
    ok, encoded = cv2.imencode(".jpg", depth_u8,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return False
    jpeg = encoded.tobytes()
    sys.stdout.buffer.write(struct.pack("<II", _DEPTH_FRAME_SENTINEL, len(jpeg)))
    sys.stdout.buffer.write(jpeg)
    sys.stdout.buffer.flush()
    return True


# -----------------------------------------------------------------------------
# Prompt setter (mirrors streamdiffusion_api.py patch)
# -----------------------------------------------------------------------------

def _patch_pipeline_set_prompt():
    if hasattr(Pipeline, "set_prompt"):
        return

    def set_prompt(self, prompt):
        import torch
        with torch.no_grad():
            ti = self._tokenizer(
                prompt,
                padding="max_length",
                max_length=self._tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            embeds = (
                self._text_encoder(ti.input_ids.to("mps"))[0]
                .cpu()
                .to(torch.float16)
                .numpy()
            )
        self._all_prompts = [prompt]
        self._all_embeds = [embeds]
        self._prompt_index = 0
        self._target_embeds = embeds
        self._current_prompt = prompt
        return prompt

    Pipeline.set_prompt = set_prompt


_patch_pipeline_set_prompt()


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="StreamDiffusion inference worker")
    parser.add_argument("--prompt", type=str,
                        default="oil painting style, masterpiece, highly detailed")
    parser.add_argument("--model", type=str, default="sdxs", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--render-size", type=int, default=512, choices=[320, 384, 512, 768])
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--feedback", type=float, default=0.1)
    parser.add_argument("--coreml-dir", type=str, default=COREML_DIR)
    parser.add_argument("--depth-backend", type=str, default=None,
                        choices=["auto", "coreml", "pytorch"],
                        help="Enable RGBD output using the specified depth backend")
    parser.add_argument("--depth-coreml-path", type=str, default=None,
                        help="Path to CoreML depth model (used with --depth-backend=coreml)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

    print(f"Inference worker PID {os.getpid()} starting", file=sys.stderr, flush=True)

    if args.depth_backend:
        try:
            from camera_rgbd import RGBDPipeline
            pipeline = RGBDPipeline(
                depth_model="auto",
                depth_backend=args.depth_backend,
                depth_coreml_path=args.depth_coreml_path,
                model_name=args.model,
                render_size=args.render_size,
                output_size=args.output_size,
                prompt=args.prompt,
                strength=args.strength,
                prompts=None,
                latent_feedback=args.feedback,
                coreml_dir=args.coreml_dir,
            )
            use_rgbd = True
        except Exception as e:
            print(f"Failed to import RGBDPipeline: {e}", file=sys.stderr, flush=True)
            sys.exit(1)
    else:
        pipeline = Pipeline(
            model_name=args.model,
            render_size=args.render_size,
            output_size=args.output_size,
            prompt=args.prompt,
            strength=args.strength,
            prompts=None,
            latent_feedback=args.feedback,
            coreml_dir=args.coreml_dir,
        )
        use_rgbd = False

    print("Inference worker ready", file=sys.stderr, flush=True)

    # Emit ready packet on stdout. jpeg_size == 0 signals readiness and carries
    # the OS PID and main thread ID so the parent can identify this instance.
    sys.stdout.buffer.write(
        struct.pack("<III", 0, os.getpid(), threading.current_thread().ident)
    )
    sys.stdout.buffer.flush()

    while True:
        packet = read_packet()
        if packet is None:
            print("STDIN closed, exiting", flush=True)
            break

        kind, payload = packet
        if kind == "prompt":
            pipeline.set_prompt(payload)
            print(f"Prompt updated: {payload}", file=sys.stderr, flush=True)
            continue

        # kind == "frame"
        frame = payload

        if use_rgbd and hasattr(pipeline, "process_frame_rgbd"):
            # process_frame_rgbd expects BGR input
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            result_bgr, depth_u8, _rgbd = pipeline.process_frame_rgbd(frame_bgr)
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            write_frame(result_rgb)
            write_depth_frame(depth_u8)
        else:
            result = pipeline.process_frame_rgb(frame)
            write_frame(result)


if __name__ == "__main__":
    main()
