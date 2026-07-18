#!/usr/bin/env python3
"""
StreamDiffusion for Mac — NDI I/O Integration with RGBD depth output
NDI Input → CoreML img2img + Depth Estimation → NDI Output (color + depth)

Usage:
    python camera_ndi.py --prompt "oil painting style"
    python camera_ndi.py --prompt "cyberpunk" --ndi-source "OBS" --ndi-output "SD-Render"
    python camera_ndi.py --prompt "watercolor" --depth-model da3-base --no-preview
"""
import os
import sys
import time
import json
import argparse
import threading
import signal

import numpy as np
import cv2
import NDIlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipelines.rgbd import RGBDPipeline
from pipelines.coreml import COREML_DIR
from configs import DEFAULT_PROMPTS, MODEL_CONFIGS
from utils.ndi import _create_ndi_sender, _send_ndi
from utils.cv2_helper import destroy_cv_windows


__all__ = ["RGBDPipeline", "NDIApp"]


class NDIApp:
    """NDI RGBD app: NDI receive, AI inference + depth, NDI send + preview."""

    def __init__(self, pipeline, ndi_source_name=None,
                 ndi_output_name="StreamDiffusion-Mac",
                 show_preview=True, blend_ratio=0.0, ema_alpha=0.85,
                 depth_preview_mode="mono"):
        self.pipeline = pipeline
        self.ndi_source_name = ndi_source_name
        self.ndi_output_name = ndi_output_name
        self.show_preview = show_preview
        self.blend_ratio = blend_ratio
        self.ema_alpha = ema_alpha
        self.depth_preview_mode = depth_preview_mode
        self.running = False

        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._result_lock = threading.Lock()
        self._latest_ai_result = None
        self._latest_depth = None
        self._ai_update_count = 0
        self._ai_fps = 0.0
        self._ema_result = None

        self._receiver = None
        self._color_sender = None
        self._depth_sender = None

    def find_ndi_sources(self, timeout_ms=3000):
        """Return a list of available NDI source names."""
        if not NDIlib.initialize():
            print("ERROR: NDI initialization failed. Is NDI SDK installed?")
            return []

        finder = NDIlib.find_create_v2()
        if not finder:
            return []

        sources = []
        waited = 0
        step = 200
        while waited < timeout_ms:
            NDIlib.find_wait_for_sources(finder, step)
            waited += step
            sources = NDIlib.find_get_current_sources(finder)
            if sources:
                break

        names = [s.ndi_name for s in sources]
        NDIlib.find_destroy(finder)
        return names

    def _find_ndi_source(self, timeout_ms=10000):
        """Find NDI source by name or auto-select."""
        if not NDIlib.initialize():
            print("ERROR: NDI initialization failed. Is NDI SDK installed?")
            return None

        finder = NDIlib.find_create_v2()
        if not finder:
            print("ERROR: Failed to create NDI finder")
            return None

        print(f"Searching for NDI sources (timeout: {timeout_ms}ms)...")
        sources = []
        waited = 0
        step = 500
        while waited < timeout_ms:
            NDIlib.find_wait_for_sources(finder, step)
            waited += step
            sources = NDIlib.find_get_current_sources(finder)
            if sources:
                break

        if not sources:
            print("No NDI sources found.")
            NDIlib.find_destroy(finder)
            return None

        print(f"Found {len(sources)} NDI source(s):")
        for i, s in enumerate(sources):
            print(f"  [{i}] {s.ndi_name}")

        selected = None
        if self.ndi_source_name:
            for s in sources:
                if self.ndi_source_name.lower() in s.ndi_name.lower():
                    selected = s
                    print(f"Auto-selected: {s.ndi_name}")
                    break
            if selected is None:
                print(f"Name '{self.ndi_source_name}' not found, using first source.")
                selected = sources[0]
        else:
            if len(sources) == 1:
                selected = sources[0]
                print(f"Selected: {selected.ndi_name}")
            else:
                try:
                    idx = int(input(f"Select NDI source [0-{len(sources)-1}]: "))
                    selected = sources[idx] if 0 <= idx < len(sources) else sources[0]
                except (ValueError, IndexError):
                    selected = sources[0]
                print(f"Selected: {selected.ndi_name}")

        NDIlib.find_destroy(finder)
        return selected

    def _create_ndi_receiver(self, source):
        """Create NDI receiver connected to source."""
        recv_settings = NDIlib.RecvCreateV3()
        recv_settings.source_to_connect_to = source
        recv_settings.color_format = NDIlib.RecvColorFormat.RECV_COLOR_FORMAT_BGRX_BGRA
        recv_settings.bandwidth = NDIlib.RecvBandwidth.RECV_BANDWIDTH_HIGHEST
        recv_settings.allow_video_fields = False

        receiver = NDIlib.recv_create_v3(recv_settings)
        if not receiver:
            print("ERROR: Failed to create NDI receiver")
            return None

        print("NDI receiver connected")
        return receiver

    @staticmethod
    def _ndi_to_bgr(video_frame):
        """Convert NDI BGRX frame to OpenCV BGR."""
        if video_frame.data is None or video_frame.data.size == 0:
            return None

        h = video_frame.yres
        w = video_frame.xres
        if h <= 0 or w <= 0:
            return None

        if video_frame.data.ndim == 1:
            stride = video_frame.line_stride_in_bytes
            if stride == 0:
                stride = w * 4
            expected = h * stride
            if video_frame.data.size < expected:
                return None
            flat = video_frame.data[:expected]
            img = flat.reshape(h, stride)
            if stride > w * 4:
                img = img[:, :w * 4]
            img = img.reshape(h, w, 4)
        elif video_frame.data.ndim == 2:
            img = video_frame.data.reshape(h, -1)
            if img.shape[1] >= w * 4:
                img = img[:, :w * 4].reshape(h, w, 4)
            else:
                return None
        elif video_frame.data.ndim == 3:
            img = video_frame.data
            if img.shape[2] >= 3:
                pass
            else:
                return None
        else:
            return None

        if img.shape[2] >= 3:
            return img[:, :, :3].copy()
        return None

    def _ndi_receive_thread(self, receiver):
        """Receive frames from NDI source."""
        while self.running:
            frame_type, video_frame, audio_frame, metadata_frame = NDIlib.recv_capture_v3(
                receiver, timeout_in_ms=100
            )

            if frame_type == NDIlib.FrameType.FRAME_TYPE_VIDEO:
                if video_frame and video_frame.data is not None and video_frame.data.size > 0:
                    frame = self._ndi_to_bgr(video_frame)
                    if frame is not None:
                        with self._frame_lock:
                            self._latest_frame = frame
                try:
                    NDIlib.recv_free_video_v2(receiver, video_frame)
                except Exception:
                    pass
            elif frame_type == NDIlib.FrameType.FRAME_TYPE_NONE:
                time.sleep(0.001)
            elif frame_type == NDIlib.FrameType.FRAME_TYPE_ERROR:
                print("NDI receive error")
                break

    def _inference_thread(self):
        """Run CoreML inference + depth on received frames."""
        fps_times = []
        while self.running:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()
            result_bgr, depth_u8, rgbd = self.pipeline.process_frame_rgbd(frame)
            elapsed = time.perf_counter() - t0

            fps_times.append(elapsed)
            if len(fps_times) > 30:
                fps_times.pop(0)

            with self._result_lock:
                self._latest_ai_result = result_bgr
                self._latest_depth = depth_u8
                self._ai_update_count += 1
                self._ai_fps = 1.0 / (sum(fps_times) / len(fps_times))

    def _stdin_thread(self):
        """Read runtime prompt/seed updates from stdin."""
        while self.running:
            try:
                line = sys.stdin.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                line = line.strip()
                if not line:
                    continue
                if line.startswith("seed:"):
                    seed_str = line[5:].strip()
                    try:
                        new_seed = int(seed_str)
                        self.pipeline.set_seed(new_seed)
                        print(f"  Seed updated: {new_seed}")
                    except ValueError:
                        print(f"  Invalid seed value: {seed_str}")
                elif line.startswith("lora:"):
                    json_str = line[5:].strip()
                    try:
                        lora_stack = json.loads(json_str)
                        if not isinstance(lora_stack, list):
                            raise ValueError("LoRA stack must be a JSON list")
                        self.pipeline.set_lora_stack(lora_stack)
                        print(f"  LoRA stack updated ({len(lora_stack)} item(s))")
                    except Exception as e:
                        print(f"  Invalid LoRA stack: {e}")
                else:
                    self.pipeline.set_prompt(line)
                    print(f"  Prompt updated: {line[:60]}")
            except Exception as e:
                if self.running:
                    print(f"  Stdin read error: {e}")

    def _make_depth_preview(self, rgb_bgr, depth):
        """Render the depth map according to the selected preview mode."""
        if depth is None:
            return np.zeros_like(rgb_bgr)

        if self.depth_preview_mode == "mono":
            return cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

        if self.depth_preview_mode == "alpha":
            gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            alpha = depth.astype(np.float32) / 255.0
            merged = gray_3ch.astype(np.float32) * alpha[:, :, None]
            return merged.clip(0, 255).astype(np.uint8)

        if self.depth_preview_mode == "alpha_color":
            alpha = depth.astype(np.float32) / 255.0
            merged = rgb_bgr.astype(np.float32) * alpha[:, :, None]
            return merged.clip(0, 255).astype(np.uint8)

        if self.depth_preview_mode == "overlay":
            gray = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            return cv2.addWeighted(rgb_bgr, 0.5, gray, 0.5, 0)

        return cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

    def run(self):
        print(f"\nOpening NDI source '{self.ndi_source_name or '(auto)'}'...")
        source = self._find_ndi_source()
        if source is None:
            print("ERROR: Cannot find NDI source!")
            return

        receiver = self._create_ndi_receiver(source)
        if receiver is None:
            return

        # Create NDI senders for color and depth
        self._color_sender = _create_ndi_sender(f"{self.ndi_output_name}-color")
        self._depth_sender = _create_ndi_sender(f"{self.ndi_output_name}-depth")

        print(f"NDI output: {self.ndi_output_name}-color / {self.ndi_output_name}-depth")
        print(f"Blend: {self.blend_ratio:.1f} (0=AI only, 1=source only)")
        print(f"EMA: {self.ema_alpha:.2f}")
        print("Controls: q=quit  s=save  n/p=prompt  +/-=blend  e/d=EMA")
        print("")

        self.running = True

        def _signal_handler(signum, frame):
            print(f"\n[signal] Received {signum}, shutting down...")
            self.running = False

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        ndi_t = threading.Thread(target=self._ndi_receive_thread, args=(receiver,), daemon=True)
        inf_t = threading.Thread(target=self._inference_thread, daemon=True)
        stdin_t = threading.Thread(target=self._stdin_thread, daemon=True)
        ndi_t.start()
        inf_t.start()
        stdin_t.start()

        out_h, out_w = self.pipeline.output_height, self.pipeline.output_width
        if out_h is None or out_w is None:
            out_w = out_h = self.pipeline.output_size

        win = f"StreamDiffusion-NDI ({self.pipeline.model_name} {self.pipeline.render_size})"
        if self.show_preview:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, out_w * 2 + 20, out_h)

        while self.running:
            with self._result_lock:
                ai_result = self._latest_ai_result
                depth = self._latest_depth
                ai_fps = self._ai_fps

            if ai_result is not None:
                if self._ema_result is None:
                    self._ema_result = ai_result.astype(np.float32)
                else:
                    self._ema_result = (
                        self.ema_alpha * self._ema_result
                        + (1.0 - self.ema_alpha) * ai_result.astype(np.float32)
                    )
                smooth_ai = self._ema_result.clip(0, 255).astype(np.uint8)

                if self.blend_ratio > 0.01:
                    # We don't have original source frame in result lock; use latest
                    with self._frame_lock:
                        source_frame = self._latest_frame
                    if source_frame is not None:
                        source_display = cv2.resize(source_frame, (out_w, out_h))
                        result_display = cv2.addWeighted(
                            smooth_ai, 1.0 - self.blend_ratio,
                            source_display, self.blend_ratio, 0,
                        )
                    else:
                        result_display = smooth_ai
                else:
                    result_display = smooth_ai

                depth_color = self._make_depth_preview(result_display, depth)
            else:
                time.sleep(0.001)
                continue

            # --- NDI Output ---
            if self._color_sender:
                _send_ndi(self._color_sender, result_display)
            if self._depth_sender and depth is not None:
                depth_bgr = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
                _send_ndi(self._depth_sender, depth_bgr)

            if self.show_preview:
                display = np.hstack([result_display, depth_color])
                pidx = self.pipeline._prompt_index + 1
                ptotal = len(self.pipeline._all_prompts)
                cv2.putText(display, f"AI: {ai_fps:.1f} FPS | Prompt {pidx}/{ptotal}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(display, self.pipeline._current_prompt[:70],
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)
                cv2.putText(display, "AI Output",
                            (out_w // 2 - 55, out_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(display, f"Depth ({self.depth_preview_mode})",
                            (out_w + out_w // 2 - 65, out_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow(win, display)

                try:
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        self.running = False
                        break
                except cv2.error:
                    self.running = False
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('s'):
                    ts = int(time.time())
                    fn_vis = f"capture_ndi_{ts}.png"
                    cv2.imwrite(fn_vis, np.hstack([result_display, depth_color]))
                    print(f"  Saved vis:  {fn_vis}")
                elif key == ord('n'):
                    p = self.pipeline.next_prompt()
                    print(f"  [{pidx}/{ptotal}] {p[:60]}")
                elif key == ord('p'):
                    p = self.pipeline.prev_prompt()
                    print(f"  [{pidx}/{ptotal}] {p[:60]}")
                elif key in (ord('+'), ord('=')):
                    self.blend_ratio = min(1.0, self.blend_ratio + 0.05)
                    print(f"  Blend: {self.blend_ratio:.2f}")
                elif key == ord('-'):
                    self.blend_ratio = max(0.0, self.blend_ratio - 0.05)
                    print(f"  Blend: {self.blend_ratio:.2f}")
                elif key == ord('e'):
                    self.ema_alpha = min(0.99, self.ema_alpha + 0.05)
                    print(f"  EMA: {self.ema_alpha:.2f}")
                elif key == ord('d'):
                    self.ema_alpha = max(0.0, self.ema_alpha - 0.05)
                    print(f"  EMA: {self.ema_alpha:.2f}")

        print("\nShutting down NDI...")
        self.running = False
        if receiver:
            NDIlib.recv_destroy(receiver)
        if self._color_sender:
            NDIlib.send_destroy(self._color_sender)
        if self._depth_sender:
            NDIlib.send_destroy(self._depth_sender)
        NDIlib.destroy()
        destroy_cv_windows()


def main():
    parser = argparse.ArgumentParser(description="StreamDiffusion for Mac — NDI I/O with RGBD")
    parser.add_argument("--prompt", type=str, default="oil painting style, masterpiece, highly detailed")
    parser.add_argument("--prompts", action="store_true", help="Use built-in prompt gallery")
    parser.add_argument("--model", type=str, default="sdxs", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--render-size", type=int, default=512, choices=[320, 384, 512, 768])
    parser.add_argument("--output-size", type=str, default="512",
                        help="Output resolution: integer (square) or WxH")
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--blend", type=float, default=0.0,
                        help="Source blend (0.0=AI only, 0.3=30%% source)")
    parser.add_argument("--ema", type=float, default=0.4, help="EMA smoothing")
    parser.add_argument("--feedback", type=float, default=0.1, help="Latent feedback")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for fixed noise")
    # Depth args
    parser.add_argument("--depth-model", type=str, default="auto",
                        choices=["auto", "da3-small", "da3-base", "da3-large",
                                 "da2-small", "da2-base", "da2-large"],
                        help="Depth estimation model")
    parser.add_argument("--depth-backend", type=str, default="auto",
                        choices=["auto", "coreml", "pytorch"],
                        help="Depth inference backend")
    parser.add_argument("--depth-coreml-path", type=str, default=None,
                        help="Path to converted CoreML depth model")
    parser.add_argument("--depth-preview-mode", type=str, default="mono",
                        choices=["mono", "alpha", "alpha_color", "overlay"],
                        help="Depth preview mode")
    # NDI args
    parser.add_argument("--ndi-source", type=str, default=None,
                        help="NDI source name (partial match). Auto-detect if not set.")
    parser.add_argument("--ndi-output", type=str, default="StreamDiffusion-Mac",
                        help="NDI output source name base (creates -color and -depth)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable OpenCV preview window")
    parser.add_argument("--coreml-dir", type=str, default=COREML_DIR)
    args = parser.parse_args()

    # Resolve output size
    output_size = args.output_size.strip().lower()
    if "x" in output_size:
        output_width, output_height = map(int, output_size.split("x"))
    else:
        output_width = output_height = int(output_size)

    print("=" * 60)
    print(f"StreamDiffusion for Mac — NDI I/O RGBD ({args.model} {args.render_size}x{args.render_size})")
    print(f"  Output: {output_width}x{output_height}")
    print("  CoreML img2img + Depth Anything RGBD generation")
    print("=" * 60)

    prompts = DEFAULT_PROMPTS if args.prompts else None

    pipeline = RGBDPipeline(
        depth_model=args.depth_model,
        depth_backend=args.depth_backend,
        depth_coreml_path=args.depth_coreml_path,
        model_name=args.model,
        render_size=args.render_size,
        output_size=output_width if output_width == output_height else args.render_size,
        output_width=output_width,
        output_height=output_height,
        prompt=args.prompt,
        strength=args.strength,
        prompts=prompts,
        latent_feedback=args.feedback,
        coreml_dir=args.coreml_dir,
        seed=args.seed,
    )

    app = NDIApp(
        pipeline=pipeline,
        ndi_source_name=args.ndi_source,
        ndi_output_name=args.ndi_output,
        show_preview=not args.no_preview,
        blend_ratio=args.blend,
        ema_alpha=args.ema,
        depth_preview_mode=args.depth_preview_mode,
    )
    app.run()


if __name__ == "__main__":
    main()
