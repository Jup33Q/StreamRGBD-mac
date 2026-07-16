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
import argparse

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipelines.coreml import Pipeline, COREML_DIR
from apps.ndi import NDIApp
from configs import DEFAULT_PROMPTS, MODEL_CONFIGS

# Backward-compatible re-exports.
__all__ = ["Pipeline", "COREML_DIR", "NDIApp"]


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
