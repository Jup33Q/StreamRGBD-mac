#!/usr/bin/env python3
"""
StreamDiffusion for Mac — RGBD Output Pipeline

Runs the CoreML img2img pipeline, then estimates depth on the AI output
and concatenates RGB + depth into a 4-channel RGBD frame.

Three depth backends are supported (in order of preference on macOS):

1. CoreML DA3-Small  : load a pre-converted `da3_small.mlpackage`
                       (fastest on Apple Silicon, see README for conversion).
2. PyTorch DA3-Small : uses the official `depth_anything_3` package.
3. PyTorch DA2-Small : Hugging Face Transformers pipeline, works out-of-the-box
                       on macOS when DA3 is not available.

The RGBD frame is exposed as H x W x 4 uint8 (RGB in first 3 channels,
depth as the 4th / alpha channel).  The preview window shows the AI RGB
output next to a color-mapped depth visualization.

Usage:
    python camera_rgbd.py --prompt "oil painting style, masterpiece"
    python camera_rgbd.py --depth-backend coreml --depth-coreml-path ./da3_small.mlpackage
    python camera_rgbd.py --depth-model da2-small --prompt "watercolor"

Controls:
    q     : quit
    s     : save current RGBD frame + depth visualization
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
import torch

# Re-use the CoreML img2img pipeline from the original camera.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera import Pipeline, DEFAULT_PROMPTS, MODEL_CONFIGS, COREML_DIR


def _default_device():
    """Pick MPS on Apple Silicon, CUDA if present, otherwise CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _normalize_depth(depth):
    """Normalize a float depth map to uint8."""
    lo, hi = depth.min(), depth.max()
    if hi - lo < 1e-6:
        return np.zeros_like(depth, dtype=np.uint8)
    norm = (depth - lo) / (hi - lo)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


class CoreMLDepthEstimator:
    """Depth Anything 3 Small running as a CoreML model (macOS/Apple Silicon)."""

    INPUT_SIZE = 504
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, model_path):
        import coremltools as ct

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CoreML depth model not found: {model_path}")

        print(f"  Loading CoreML depth model: {model_path}")
        self.model = ct.models.MLModel(
            model_path,
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )

        # Auto-discover input/output tensor names from the model spec.
        spec = self.model.get_spec()
        in_feat = spec.description.input[0]
        out_feat = spec.description.output[0]
        self.in_name = in_feat.name
        self.out_name = out_feat.name
        in_shape = list(in_feat.type.multiArrayType.shape)
        out_shape = list(out_feat.type.multiArrayType.shape)
        print(f"    Input : {self.in_name} -> {in_shape}")
        print(f"    Output: {self.out_name} -> {out_shape}")

    def estimate(self, rgb_np):
        """Estimate depth for a single RGB uint8 image."""
        h, w = rgb_np.shape[:2]

        # 1. Resize preserving aspect ratio and pad to INPUT_SIZE.
        scale = min(self.INPUT_SIZE / w, self.INPUT_SIZE / h)
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        resized = cv2.resize(rgb_np, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype=np.float32)
        padded[:scaled_h, :scaled_w] = resized.astype(np.float32) / 255.0

        # 2. ImageNet normalization.
        normalized = (padded - self.IMAGENET_MEAN) / self.IMAGENET_STD

        # 3. Add batch dims expected by the traced DA3 model: (1, 1, 3, H, W).
        x = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, None])

        # 4. CoreML inference.
        pred = self.model.predict({self.in_name: x})
        depth = pred[self.out_name]

        # 5. Remove batch/channel dims, crop away padding, and resize back.
        depth = np.squeeze(depth)
        if depth.ndim != 2:
            raise RuntimeError(f"Unexpected CoreML depth output shape after squeeze: {depth.shape}")

        depth_cropped = depth[:scaled_h, :scaled_w]
        depth_full = cv2.resize(depth_cropped.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return _normalize_depth(depth_full)


class TorchDepthEstimator:
    """Depth Anything V3/V2 via PyTorch / Transformers."""

    DA3_NAME = "depth-anything/DA3-SMALL"
    DA2_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self._pipe = None
        self._da3_model = None

        if model_name == "da3-small":
            self._load_da3()
        elif model_name == "da2-small":
            self._load_da2()
        else:
            raise ValueError(f"Unknown PyTorch depth model: {model_name}")

    @staticmethod
    def _da3_available():
        try:
            from depth_anything_3.api import DepthAnything3  # noqa: F401
            return True
        except Exception:
            return False

    def _load_da3(self):
        from depth_anything_3.api import DepthAnything3
        self._da3_model = DepthAnything3.from_pretrained(self.DA3_NAME).to(self.device)
        self._da3_model.eval()

    def _load_da2(self):
        from transformers import pipeline
        self._pipe = pipeline(
            "depth-estimation",
            model=self.DA2_NAME,
            device=self.device,
        )

    def estimate(self, rgb_np):
        """Estimate depth for a single RGB uint8 image."""
        h, w = rgb_np.shape[:2]
        if self.model_name == "da3-small":
            depth = self._estimate_da3(rgb_np)
        else:
            depth = self._estimate_da2(rgb_np)

        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth

    def _estimate_da3(self, rgb_np):
        with torch.no_grad():
            prediction = self._da3_model.inference([rgb_np])
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        return _normalize_depth(depth)

    def _estimate_da2(self, rgb_np):
        from PIL import Image
        pred = self._pipe(Image.fromarray(rgb_np))
        depth = np.asarray(pred["predicted_depth"], dtype=np.float32)
        return _normalize_depth(depth)


class DepthEstimator:
    """Picks the best available depth estimator for the current platform."""

    def __init__(self, model_name="auto", backend="auto", coreml_path=None, device=None):
        self.device = device or _default_device()

        # Resolve default CoreML model path.
        if coreml_path is None:
            coreml_path = os.path.join(COREML_DIR, "da3_small.mlpackage")

        # Auto-select backend.
        if backend == "auto":
            if os.path.exists(coreml_path):
                backend = "coreml"
            elif model_name in ("auto", "da3-small") and TorchDepthEstimator._da3_available():
                backend = "pytorch"
                model_name = "da3-small"
            else:
                backend = "pytorch"
                if model_name == "auto":
                    print("  Depth Anything V3/CoreML not found; using Depth Anything V2 Small.")
                model_name = "da2-small"

        self.backend = backend
        self.model_name = model_name

        if backend == "coreml":
            self._estimator = CoreMLDepthEstimator(coreml_path)
        elif backend == "pytorch":
            self._estimator = TorchDepthEstimator(model_name, self.device)
        else:
            raise ValueError(f"Unknown depth backend: {backend}")

        print(f"  Depth estimator ready: {backend} ({model_name}) on {self.device}")

    def estimate(self, rgb_np):
        return self._estimator.estimate(rgb_np)


class RGBDPipeline(Pipeline):
    """CoreML img2img pipeline extended with RGBD output."""

    def __init__(self, depth_model="auto", depth_backend="auto",
                 depth_coreml_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("\n--- Loading Depth Estimator ---")
        self.depth_estimator = DepthEstimator(
            model_name=depth_model,
            backend=depth_backend,
            coreml_path=depth_coreml_path,
        )

    def process_frame_rgbd(self, frame_bgr):
        """
        Run img2img + depth estimation and return RGB, depth, and RGBD.

        Returns:
            result_bgr: output_size x output_size x 3 uint8 BGR image.
            depth_u8:   output_size x output_size uint8 depth map.
            rgbd:       output_size x output_size x 4 uint8 RGBD image.
        """
        # 1. Run the base CoreML img2img pipeline.
        result_bgr = self.process_frame(frame_bgr)

        # 2. Estimate depth on the AI RGB output.
        rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        depth_u8 = self.depth_estimator.estimate(rgb)

        # 3. Concatenate RGB and depth into a 4-channel RGBD frame.
        rgbd = np.concatenate([rgb, depth_u8[:, :, np.newaxis]], axis=2)

        return result_bgr, depth_u8, rgbd


class RGBDCameraApp:
    """3-thread camera application that also produces RGBD frames."""

    PREVIEW_MODES = ["mono", "alpha", "alpha_color", "overlay"]

    def __init__(self, pipeline, camera_id=0, blend_ratio=0.0, ema_alpha=0.85,
                 depth_preview_mode="mono"):
        self.pipeline = pipeline
        self.camera_id = camera_id
        self.blend_ratio = blend_ratio
        self.ema_alpha = ema_alpha
        if depth_preview_mode not in self.PREVIEW_MODES:
            raise ValueError(f"Unknown preview mode: {depth_preview_mode}")
        self.depth_preview_mode = depth_preview_mode
        self.running = False

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

        out_sz = self.pipeline.output_size
        print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"Blend: {self.blend_ratio:.1f} (0=AI only, 1=camera only)")
        print(f"EMA: {self.ema_alpha:.2f}")
        print(f"Depth preview: {self.depth_preview_mode} (mono/alpha/alpha_color/overlay)")
        print("Controls: q=quit  s=save  n/p=prompt  +/-=blend  e/d=EMA  m=depth mode")
        print("")

        self.running = True
        cam_t = threading.Thread(target=self._camera_thread, args=(cap,), daemon=True)
        inf_t = threading.Thread(target=self._inference_thread, daemon=True)
        cam_t.start()
        inf_t.start()

        win = f"StreamDiffusion-RGBD ({self.pipeline.model_name} {self.pipeline.render_size})"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, out_sz * 2 + 20, out_sz)

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

            # Camera preview (square crop)
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

                depth_color = self._make_depth_preview(result_display, depth)
            else:
                result_display = cam_display
                depth_color = np.zeros_like(cam_display)

            display = np.hstack([result_display, depth_color])

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
                        (out_sz // 2 - 55, out_sz - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"Depth ({self.depth_preview_mode})",
                        (out_sz + out_sz // 2 - 65, out_sz - 10),
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

        total = time.perf_counter() - total_start
        with self._result_lock:
            ai_total = self._ai_update_count
        print(f"\nSession: {display_count} display frames in {total:.1f}s = {display_count / total:.1f} display FPS")
        print(f"  AI+Depth inference: {ai_total} frames = {ai_total / total:.1f} FPS")

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="StreamDiffusion for Mac — RGBD Output")
    parser.add_argument("--prompt", type=str, default="oil painting style, masterpiece, highly detailed")
    parser.add_argument("--prompts", action="store_true",
                        help="Use built-in prompt gallery (10 styles)")
    parser.add_argument("--model", type=str, default="sdxs", choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use (default: sdxs for best performance)")
    parser.add_argument("--render-size", type=int, default=512, choices=[320, 384, 512])
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--blend", type=float, default=0.0,
                        help="Camera blend (0.0=AI only, 0.3=30%% camera)")
    parser.add_argument("--ema", type=float, default=0.4,
                        help="EMA smoothing (0=none, 0.9=heavy)")
    parser.add_argument("--feedback", type=float, default=0.1,
                        help="Latent feedback (0=none, 0.3=30%% prev frame)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--coreml-dir", type=str, default=COREML_DIR)
    parser.add_argument("--depth-model", type=str, default="auto",
                        choices=["auto", "da3-small", "da2-small"],
                        help="PyTorch depth model (used when backend is pytorch)")
    parser.add_argument("--depth-backend", type=str, default="auto",
                        choices=["auto", "coreml", "pytorch"],
                        help=("Depth inference backend: auto prefers CoreML, "
                              "then PyTorch DA3, then PyTorch DA2"))
    parser.add_argument("--depth-coreml-path", type=str, default=None,
                        help="Path to a converted DA3-Small CoreML package "
                             "(default: <coreml-dir>/da3_small.mlpackage)")
    parser.add_argument("--depth-preview-mode", type=str, default="mono",
                        choices=["mono", "alpha", "alpha_color", "overlay"],
                        help=("Preview mode for the right pane: "
                              "mono=grayscale depth, "
                              "alpha=grayscale RGB composited with depth as alpha, "
                              "alpha_color=color RGB composited with depth as alpha, "
                              "overlay=50%% blend of RGB and grayscale depth"))
    args = parser.parse_args()

    print("=" * 60)
    print(f"StreamDiffusion for Mac — RGBD Output ({args.model} {args.render_size}x{args.render_size})")
    print("  CoreML img2img + Depth Anything RGBD generation")
    print("=" * 60)

    prompts = DEFAULT_PROMPTS if args.prompts else None

    pipeline = RGBDPipeline(
        depth_model=args.depth_model,
        depth_backend=args.depth_backend,
        depth_coreml_path=args.depth_coreml_path,
        model_name=args.model,
        render_size=args.render_size,
        output_size=args.output_size,
        prompt=args.prompt,
        strength=args.strength,
        prompts=prompts,
        latent_feedback=args.feedback,
        coreml_dir=args.coreml_dir,
    )

    app = RGBDCameraApp(
        pipeline=pipeline,
        camera_id=args.camera,
        blend_ratio=args.blend,
        ema_alpha=args.ema,
        depth_preview_mode=args.depth_preview_mode,
    )
    app.run()


if __name__ == "__main__":
    main()
