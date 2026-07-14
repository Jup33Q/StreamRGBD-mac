#!/usr/bin/env python3
"""
StreamDiffusion for Mac — NDI I/O Integration
NDI Input → CoreML img2img → NDI Output (optional preview)

Usage:
    python camera_ndi.py --prompt "oil painting style"
    python camera_ndi.py --prompt "cyberpunk" --ndi-source "OBS" --ndi-output "SD-Render"
    python camera_ndi.py --prompt "watercolor" --ndi-source "iPhone NDI" --no-preview

Controls:
    q     : quit
    s     : save current frame
    n / p : next / previous prompt
    + / - : adjust camera blend ratio
    e / d : adjust EMA smoothing
"""
import os
import sys
import time
import argparse
import threading
import numpy as np
import cv2
import NDIlib

# Import Pipeline from the original camera.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera import Pipeline, DEFAULT_PROMPTS, MODEL_CONFIGS, COREML_DIR


class NDIApp:
    """3-thread NDI app: NDI receive, AI inference, NDI send + preview."""

    def __init__(self, pipeline, ndi_source_name=None, ndi_output_name="StreamDiffusion-Mac",
                 show_preview=True, blend_ratio=0.0, ema_alpha=0.85):
        self.pipeline = pipeline
        self.ndi_source_name = ndi_source_name
        self.ndi_output_name = ndi_output_name
        self.show_preview = show_preview
        self.blend_ratio = blend_ratio
        self.ema_alpha = ema_alpha
        self.running = False

        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._result_lock = threading.Lock()
        self._latest_ai_result = None
        self._ai_update_count = 0
        self._ai_fps = 0.0
        self._ema_result = None

        self._receiver = None
        self._sender = None

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
            print("No NDI sources found. Use --no-ndi to run with webcam only.")
            NDIlib.find_destroy(finder)
            return None

        print(f"Found {len(sources)} NDI source(s):")
        for i, s in enumerate(sources):
            print(f"  [{i}] {s.ndi_name}")

        # Select source
        selected = None
        if self.ndi_source_name:
            # Find by partial name match (case-insensitive)
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

    def _create_ndi_sender(self):
        """Create NDI sender for output."""
        send_settings = NDIlib.SendCreate(p_ndi_name=self.ndi_output_name)
        sender = NDIlib.send_create(send_settings)
        if not sender:
            print("WARNING: Failed to create NDI sender")
            return None
        print(f"NDI sender created: '{self.ndi_output_name}'")
        return sender

    @staticmethod
    def _ndi_to_bgr(video_frame):
        """Convert NDI BGRX frame to OpenCV BGR."""
        if video_frame.data is None or video_frame.data.size == 0:
            return None

        h = video_frame.yres
        w = video_frame.xres
        if h <= 0 or w <= 0:
            return None

        # video_frame.data is a numpy array; shape depends on how NDIlib provides it.
        # Try to interpret as (h, w, 4) BGRX first.
        if video_frame.data.ndim == 1:
            # Flattened data: total bytes = h * line_stride
            stride = video_frame.line_stride_in_bytes
            if stride == 0:
                stride = w * 4
            expected = h * stride
            if video_frame.data.size < expected:
                return None
            # Reshape with stride and crop to width
            flat = video_frame.data[:expected]
            img = flat.reshape(h, stride)
            if stride > w * 4:
                img = img[:, :w * 4]
            img = img.reshape(h, w, 4)
        elif video_frame.data.ndim == 2:
            # (h, w*4) or (h, stride)
            img = video_frame.data.reshape(h, -1)
            if img.shape[1] >= w * 4:
                img = img[:, :w * 4].reshape(h, w, 4)
            else:
                return None
        elif video_frame.data.ndim == 3:
            img = video_frame.data
            if img.shape[2] >= 3:
                pass  # OK
            else:
                return None
        else:
            return None

        # Extract BGR (first 3 channels of BGRX)
        if img.shape[2] >= 3:
            return img[:, :, :3].copy()
        return None

    @staticmethod
    def _bgr_to_ndi(frame_bgr):
        """Convert OpenCV BGR to NDI BGRX frame."""
        h, w = frame_bgr.shape[:2]
        bgrx = np.zeros((h, w, 4), dtype=np.uint8)
        bgrx[:, :, :3] = frame_bgr
        bgrx[:, :, 3] = 255

        vf = NDIlib.VideoFrameV2()
        vf.xres = w
        vf.yres = h
        vf.FourCC = NDIlib.FourCCVideoType.FOURCC_VIDEO_TYPE_BGRX
        vf.line_stride_in_bytes = w * 4
        vf.frame_rate_N = 30
        vf.frame_rate_D = 1
        vf.data = bgrx
        return vf

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
                # recv_free_video_v2 may or may not be needed; free to be safe
                try:
                    NDIlib.recv_free_video_v2(receiver, video_frame)
                except Exception:
                    pass
            elif frame_type == NDIlib.FrameType.FRAME_TYPE_NONE:
                time.sleep(0.001)
            elif frame_type == NDIlib.FrameType.FRAME_TYPE_ERROR:
                print("NDI receive error")
                break
            # Audio/metadata: ignore

    def _inference_thread(self):
        """Run CoreML inference on received frames."""
        fps_times = []
        while self.running:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()
            result = self.pipeline.process_frame(frame)
            elapsed = time.perf_counter() - t0

            fps_times.append(elapsed)
            if len(fps_times) > 30:
                fps_times.pop(0)

            with self._result_lock:
                self._latest_ai_result = result
                self._ai_update_count += 1
                self._ai_fps = 1.0 / (sum(fps_times) / len(fps_times))

    def run(self):
        """Main loop: NDI setup, threads, display/sending."""
        # --- Find and connect NDI source ---
        source = self._find_ndi_source()
        self._receiver = self._create_ndi_receiver(source) if source else None

        # --- Create NDI sender ---
        self._sender = self._create_ndi_sender()

        # --- Start threads ---
        self.running = True
        threads = []

        if self._receiver:
            ndi_t = threading.Thread(target=self._ndi_receive_thread,
                                     args=(self._receiver,), daemon=True)
            ndi_t.start()
            threads.append(ndi_t)
        else:
            print("WARNING: No NDI receiver. Running with no input.")

        inf_t = threading.Thread(target=self._inference_thread, daemon=True)
        inf_t.start()
        threads.append(inf_t)

        out_sz = self.pipeline.output_size
        print(f"\nNDI: {self.ndi_source_name or 'Auto'} → CoreML → {self.ndi_output_name}")
        print(f"Blend: {self.blend_ratio:.1f} (0=AI only, 1=camera/NDI only)")
        print(f"EMA: {self.ema_alpha:.2f}")
        print("Controls: q=quit  s=save  n/p=prompt  +/-=blend  e/d=EMA")
        print("")

        # --- Preview / NDI send loop ---
        if self.show_preview:
            cv2.namedWindow("StreamDiffusion-NDI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("StreamDiffusion-NDI", out_sz * 2 + 20, out_sz)

        display_count = 0
        total_start = time.perf_counter()
        display_fps_times = []
        window_title = "StreamDiffusion-NDI"

        while self.running:
            dt0 = time.perf_counter()

            with self._frame_lock:
                frame = self._latest_frame
            with self._result_lock:
                ai_result = self._latest_ai_result
                ai_fps = self._ai_fps

            if frame is None:
                time.sleep(0.001)
                continue

            # Prepare display frames
            h, w = frame.shape[:2]
            if w > h:
                off = (w - h) // 2
                fsq = frame[:, off:off + h]
            elif h > w:
                off = (h - w) // 2
                fsq = frame[off:off + w, :]
            else:
                fsq = frame
            cam_display = cv2.resize(fsq, (out_sz, out_sz))

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
                    result_display = cv2.addWeighted(
                        smooth_ai, 1.0 - self.blend_ratio,
                        cam_display, self.blend_ratio, 0,
                    )
                else:
                    result_display = smooth_ai
            else:
                result_display = cam_display

            # --- NDI Output ---
            if self._sender:
                try:
                    ndi_frame = self._bgr_to_ndi(result_display)
                    NDIlib.send_send_video_v2(self._sender, ndi_frame)
                except Exception as e:
                    print(f"NDI send error: {e}")

            # --- Preview ---
            if self.show_preview:
                display = np.hstack([cam_display, result_display])
                display_count += 1
                dt = time.perf_counter() - dt0
                display_fps_times.append(dt)
                if len(display_fps_times) > 60:
                    display_fps_times.pop(0)

                pidx = self.pipeline._prompt_index + 1
                ptotal = len(self.pipeline._all_prompts)
                ptext = self.pipeline._current_prompt[:70]

                cv2.putText(display, f"AI: {ai_fps:.1f} FPS | Prompt {pidx}/{ptotal}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(display, ptext,
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)
                cv2.putText(display, "NDI Input",
                            (out_sz // 2 - 40, out_sz - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(display, "AI Output",
                            (out_sz + out_sz // 2 - 55, out_sz - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow(window_title, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('s'):
                    fn = f"capture_{int(time.time())}.png"
                    cv2.imwrite(fn, result_display)
                    print(f"  Saved: {fn}")
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
            else:
                time.sleep(0.001)

        # --- Cleanup ---
        self.running = False
        for t in threads:
            t.join(timeout=2)

        total = time.perf_counter() - total_start
        with self._result_lock:
            ai_total = self._ai_update_count
        avg_fps = ai_total / total if total > 0 else 0
        print(f"\nSession: {display_count} display frames in {total:.1f}s")
        print(f"  AI inference: {ai_total} frames = {avg_fps:.1f} AI FPS")

        if self._receiver:
            try:
                NDIlib.recv_destroy(self._receiver)
            except Exception:
                pass
        if self._sender:
            try:
                NDIlib.send_destroy(self._sender)
            except Exception:
                pass
        try:
            NDIlib.destroy()
        except Exception:
            pass

        if self.show_preview:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="StreamDiffusion for Mac — NDI I/O")
    parser.add_argument("--prompt", type=str, default="oil painting style, masterpiece, highly detailed")
    parser.add_argument("--prompts", action="store_true", help="Use built-in prompt gallery")
    parser.add_argument("--model", type=str, default="sdxs", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--render-size", type=int, default=512, choices=[320, 384, 512])
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--blend", type=float, default=0.0,
                        help="Camera blend (0.0=AI only, 0.3=30%% camera)")
    parser.add_argument("--ema", type=float, default=0.4, help="EMA smoothing")
    parser.add_argument("--feedback", type=float, default=0.1, help="Latent feedback")
    # NDI args
    parser.add_argument("--ndi-source", type=str, default=None,
                        help="NDI source name (partial match). Auto-detect if not set.")
    parser.add_argument("--ndi-output", type=str, default="StreamDiffusion-Mac",
                        help="NDI output source name")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable OpenCV preview window (headless NDI processing)")
    parser.add_argument("--coreml-dir", type=str, default=COREML_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print(f"StreamDiffusion for Mac — NDI I/O ({args.model} {args.render_size}x{args.render_size})")
    print("  CoreML-accelerated NDI video processing")
    print("=" * 60)

    prompts = DEFAULT_PROMPTS if args.prompts else None

    pipeline = Pipeline(
        model_name=args.model,
        render_size=args.render_size,
        output_size=args.output_size,
        prompt=args.prompt,
        strength=args.strength,
        prompts=prompts,
        latent_feedback=args.feedback,
        coreml_dir=args.coreml_dir,
    )

    app = NDIApp(
        pipeline=pipeline,
        ndi_source_name=args.ndi_source,
        ndi_output_name=args.ndi_output,
        show_preview=not args.no_preview,
        blend_ratio=args.blend,
        ema_alpha=args.ema,
    )
    app.run()


if __name__ == "__main__":
    main()
