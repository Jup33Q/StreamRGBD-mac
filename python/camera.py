#!/usr/bin/env python3
"""
StreamDiffusion for Mac — Real-time Camera img2img Pipeline

CoreML-accelerated real-time image-to-image transformation using
diffusion models on Apple Silicon. Achieves 22.7 FPS on M3 Ultra
with SDXS-512 at 512x512 resolution.

Architecture:
  - Camera thread:    captures frames continuously
  - Inference thread: runs full CoreML pipeline (VAE enc → UNet → VAE dec)
  - Display thread:   blends camera + AI output at 30+ FPS

Usage:
    python camera.py --prompt "oil painting style, masterpiece"
    python camera.py --prompt "watercolor painting" --blend 0.3
    python camera.py --model sd-turbo --prompt "anime style"

Controls:
    q     : quit
    s     : save current frame
    n / p : next / previous prompt
    + / - : adjust camera blend ratio
    e / d : adjust EMA smoothing
"""
import os
import sys
import argparse

# Allow running this wrapper directly from the python/ directory.
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipelines.coreml import Pipeline, COREML_DIR
from configs import MODEL_CONFIGS, DEFAULT_PROMPTS
from apps.camera import CameraApp

# Backward-compatible re-exports.
__all__ = ["Pipeline", "CameraApp", "COREML_DIR", "MODEL_CONFIGS", "DEFAULT_PROMPTS"]


def main():
    parser = argparse.ArgumentParser(description="StreamDiffusion for Mac — Real-time Camera")
    parser.add_argument("--prompt", type=str, default="oil painting style, masterpiece, highly detailed")
    parser.add_argument("--prompts", action="store_true",
                        help="Use built-in prompt gallery (10 styles)")
    parser.add_argument("--model", type=str, default="sdxs", choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use (default: sdxs for best performance). "
                             "sd-1-5 requires the CoreML UNet to be converted first.")
    parser.add_argument("--render-size", type=int, default=512, choices=[320, 384, 512])
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--blend", type=float, default=0.0,
                        help="Camera blend (0.0=AI only, 0.3=30%% camera)")
    parser.add_argument("--ema", type=float, default=0.4,
                        help="EMA smoothing (0=none, 0.9=heavy)")
    parser.add_argument("--feedback", type=float, default=0.1,
                        help="Latent feedback (0=none, 0.3=30%% prev frame)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for fixed noise (changes image texture)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--coreml-dir", type=str, default=COREML_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print(f"StreamDiffusion for Mac — {args.model} {args.render_size}x{args.render_size}")
    print("  CoreML-accelerated real-time camera img2img")
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
        seed=args.seed,
    )

    app = CameraApp(
        pipeline=pipeline,
        camera_id=args.camera,
        blend_ratio=args.blend,
        ema_alpha=args.ema,
    )
    app.run()


if __name__ == "__main__":
    main()
