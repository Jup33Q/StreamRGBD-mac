#!/usr/bin/env python3
"""RGBD camera application with optional NDI output."""
import os
import sys
import time
import json
import threading

import numpy as np
import cv2

from utils.ndi import _check_ndi, _create_ndi_sender, _send_ndi
from pipelines.rgbd import _crop_to_aspect


class RGBDCameraApp:
    """3-thread camera application that also produces RGBD frames, with optional NDI output."""

    PREVIEW_MODES = ["mono", "alpha", "alpha_color", "overlay"]

    def __init__(self, pipeline, camera_id=0, blend_ratio=0.0, ema_alpha=0.85,
                 depth_preview_mode="mono", ndi_output_name=None,
                 output_width=None, output_height=None):
        self.pipeline = pipeline
        self.camera_id = camera_id
        self.blend_ratio = blend_ratio
        self.ema_alpha = ema_alpha
        if depth_preview_mode not in self.PREVIEW_MODES:
            raise ValueError(f"Unknown preview mode: {depth_preview_mode}")
        self.depth_preview_mode = depth_preview_mode
        self.running = False
        self.ndi_output_name = ndi_output_name
        self._color_sender = None
        self._depth_sender = None
        self.output_width = output_width or getattr(pipeline, "output_width", pipeline.output_size)
        self.output_height = output_height or getattr(pipeline, "output_height", pipeline.output_size)

        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._result_lock = threading.Lock()
        self._latest_ai_result = None
        self._latest_depth = None
        self._latest_rgbd = None
        self._ai_update_count = 0
        self._ai_fps = 0.0
        self._ema_result = None

    def _camera_thread(self, cap):
        while self.running:
            ret, frame = cap.read()
            if ret:
                with self._frame_lock:
                    self._latest_frame = frame

    def _stdin_thread(self):
        """Read new prompts, seed updates and LoRA stack updates from stdin while running."""
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
            # Grayscale RGB with depth used as alpha (composited on black).
            gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            alpha = depth.astype(np.float32) / 255.0
            merged = gray_3ch.astype(np.float32) * alpha[:, :, None]
            return merged.clip(0, 255).astype(np.uint8)

        if self.depth_preview_mode == "alpha_color":
            # Color RGB with depth used as alpha (composited on black).
            alpha = depth.astype(np.float32) / 255.0
            merged = rgb_bgr.astype(np.float32) * alpha[:, :, None]
            return merged.clip(0, 255).astype(np.uint8)

        if self.depth_preview_mode == "overlay":
            # 50/50 blend of RGB and grayscale depth.
            gray = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            return cv2.addWeighted(rgb_bgr, 0.5, gray, 0.5, 0)

    def _inference_thread(self):
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
                self._latest_rgbd = rgbd
                self._ai_update_count += 1
                self._ai_fps = 1.0 / (sum(fps_times) / len(fps_times))

    def run(self):
        print(f"\nOpening camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("ERROR: Cannot open camera!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)

        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None:
                break
            time.sleep(0.1)
        else:
            print("ERROR: Cannot read frames!")
            cap.release()
            return

        out_w, out_h = self.output_width, self.output_height
        print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"Output: {out_w}x{out_h}")
        print(f"Blend: {self.blend_ratio:.1f} (0=AI only, 1=camera only)")
        print(f"EMA: {self.ema_alpha:.2f}")
        print(f"Depth preview: {self.depth_preview_mode} (mono/alpha/alpha_color/overlay)")
        if self.ndi_output_name:
            if _check_ndi():
                self._color_sender = _create_ndi_sender(f"{self.ndi_output_name}-color")
                self._depth_sender = _create_ndi_sender(f"{self.ndi_output_name}-depth")
            else:
                print("WARNING: NDIlib not installed; NDI output disabled.")
        print("Controls: q=quit  s=save  n/p=prompt  +/-=blend  e/d=EMA  m=depth mode")
        print("")

        self.running = True
        cam_t = threading.Thread(target=self._camera_thread, args=(cap,), daemon=True)
        inf_t = threading.Thread(target=self._inference_thread, daemon=True)
        stdin_t = threading.Thread(target=self._stdin_thread, daemon=True)
        cam_t.start()
        inf_t.start()
        stdin_t.start()

        win = f"StreamDiffusion-RGBD ({self.pipeline.model_name} {self.pipeline.render_size})"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, out_w * 2 + 20, out_h)

        display_count = 0
        total_start = time.perf_counter()
        display_fps_times = []

        while self.running:
            dt0 = time.perf_counter()

            with self._frame_lock:
                frame = self._latest_frame
            with self._result_lock:
                ai_result = self._latest_ai_result
                depth = self._latest_depth
                rgbd = self._latest_rgbd
                ai_fps = self._ai_fps

            if frame is None:
                time.sleep(0.001)
                continue

            # Camera preview: crop to the same aspect ratio as the output.
            cam_display = _crop_to_aspect(frame, out_w, out_h)

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

                depth_color = self._make_depth_preview(result_display, depth)
            else:
                result_display = cam_display
                depth_color = np.zeros_like(cam_display)

            display = np.hstack([result_display, depth_color])

            # --- NDI Output ---
            if self._color_sender:
                _send_ndi(self._color_sender, result_display)
            if self._depth_sender and depth is not None:
                depth_bgr = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
                _send_ndi(self._depth_sender, depth_bgr)

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
            cv2.putText(display, "AI Output",
                        (out_w // 2 - 55, out_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"Depth ({self.depth_preview_mode})",
                        (out_w + out_w // 2 - 65, out_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow(win, display)

            # Allow closing the window with the red X button (macOS/Linux/Windows).
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
                if rgbd is not None:
                    fn_rgbd = f"capture_rgbd_{ts}.png"
                    cv2.imwrite(fn_rgbd, cv2.cvtColor(rgbd, cv2.COLOR_RGBA2BGRA))
                    print(f"  Saved RGBD: {fn_rgbd}")
                fn_vis = f"capture_depthvis_{ts}.png"
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
            elif key == ord('m'):
                idx = self.PREVIEW_MODES.index(self.depth_preview_mode)
                self.depth_preview_mode = self.PREVIEW_MODES[(idx + 1) % len(self.PREVIEW_MODES)]
                print(f"  Depth preview: {self.depth_preview_mode}")

        self.running = False
        cam_t.join(timeout=2)
        inf_t.join(timeout=2)
        stdin_t.join(timeout=1)

        total = time.perf_counter() - total_start
        with self._result_lock:
            ai_total = self._ai_update_count
        print(f"\nSession: {display_count} display frames in {total:.1f}s = {display_count / total:.1f} display FPS")
        print(f"  AI+Depth inference: {ai_total} frames = {ai_total / total:.1f} FPS")

        cap.release()
        cv2.destroyAllWindows()

        if self._color_sender:
            try:
                import NDIlib
                NDIlib.send_destroy(self._color_sender)
            except Exception:
                pass
        if self._depth_sender:
            try:
                import NDIlib
                NDIlib.send_destroy(self._depth_sender)
            except Exception:
                pass
