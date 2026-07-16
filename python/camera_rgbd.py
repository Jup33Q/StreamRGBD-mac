#!/usr/bin/env python3
"""
StreamDiffusion for Mac — RGBD Output Pipeline with NDI Support

Runs the CoreML img2img pipeline, estimates depth on the AI output,
and produces dual NDI output streams: AI color (-color suffix) and depth (-depth suffix).

Three depth backends are supported (in order of preference on macOS):

1. CoreML DA3-Small  : load a pre-converted `da3_small.mlpackage`
                       (fastest on Apple Silicon, see README for conversion).
2. PyTorch DA3-Small : uses the official `depth_anything_3` package.
3. PyTorch DA2-Small : Hugging Face Transformers pipeline, works out-of-the-box
                       on macOS when DA3 is not available.

The RGBD frame is exposed as H x W x 4 uint8 (RGB in first 3 channels,
depth as the 4th / alpha channel).  The preview window shows the AI RGB
output next to a color-mapped depth visualization.

NDI Output:
    --ndi-output NAME  creates two NDI sources: "NAME-color" and "NAME-depth"
    Requires ndi-python (pip install ndi-python)

Usage:
    python camera_rgbd.py --prompt "oil painting style, masterpiece"
    python camera_rgbd.py --ndi-output "StreamDiffusion-RGBD"
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
import json
import argparse
import threading
import numpy as np
import cv2
import torch

# Python 3.12 compat: coremltools imports distutils
if "distutils" not in sys.modules:
    try:
        import setuptools
        sys.modules["distutils"] = setuptools._distutils
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipelines.coreml import Pipeline, COREML_DIR
from pipelines.rgbd import RGBDPipeline
from pipelines.torch import TorchRGBDPipeline
from apps.rgbd import RGBDCameraApp
from depth.estimators import DepthEstimator
from configs import DEFAULT_PROMPTS, MODEL_CONFIGS
from lkg_bridge import create_lkg_renderer, LKGRGBDRenderer

# Backward-compatible re-exports for scripts that import directly from this module.
__all__ = [
    "Pipeline",
    "COREML_DIR",
    "RGBDPipeline",
    "RGBDCameraApp",
    "DepthEstimator",
    "TorchRGBDPipeline",
    "_resolve_lora_path",
]


def _resolve_lora_path(path, project_dir):
    """Resolve a LoRA path to an existing file, trying multiple bases."""
    if not path:
        return path
    if os.path.isabs(path) and os.path.exists(path):
        return path
    # Try relative to project root and python/ directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for base in [project_dir, script_dir]:
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    # Try as a bare name in the loras directory
    loras_dir = os.path.join(script_dir, "loras")
    named_path = os.path.join(loras_dir, f"{path}.safetensors")
    if os.path.exists(named_path):
        return os.path.abspath(named_path)
    # Try matching the basename from the path in loras directory
    basename = os.path.basename(path)
    basename_path = os.path.join(loras_dir, basename)
    if os.path.exists(basename_path):
        return os.path.abspath(basename_path)
    return path


def _parse_output_size(value):
    """Parse --output-size as an int (square) or 'WxH' string."""
    value = str(value).strip().lower()
    if "x" in value:
        parts = value.split("x")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Invalid output size: {value!r}. Use 'WxH' or an integer.")
        try:
            return int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid output size: {value!r}. Use 'WxH' or an integer.") from exc
    try:
        size = int(value)
        return size, size
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid output size: {value!r}. Use 'WxH' or an integer.") from exc


def main():
    parser = argparse.ArgumentParser(description="StreamDiffusion for Mac — RGBD Output")
    parser.add_argument("--prompt", type=str, default="oil painting style, masterpiece, highly detailed")
    parser.add_argument("--prompts", action="store_true",
                        help="Use built-in prompt gallery (10 styles)")
    parser.add_argument("--model", type=str, default="sdxs", choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use (default: sdxs for best performance). "
                             "When --lora is used, the pipeline auto-switches to sd-1-5.")
    parser.add_argument("--render-size", type=int, default=512, choices=[320, 384, 512, 768])
    parser.add_argument("--output-size", type=str, default="512",
                        help="Output resolution: integer (square) or WxH (e.g. 720x1280).")
    parser.add_argument("--output-width", type=int, default=None,
                        help="Override output width (default: derived from --output-size).")
    parser.add_argument("--output-height", type=int, default=None,
                        help="Override output height (default: derived from --output-size).")
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
    _DEPTH_MODELS = [
        "auto",
        "da3-small", "da3-base", "da3-large",
        "da2-small", "da2-base", "da2-large",
    ]
    parser.add_argument("--depth-model", type=str, default="auto",
                        choices=_DEPTH_MODELS,
                        help="PyTorch depth model (used when backend is pytorch)")
    parser.add_argument("--depth-backend", type=str, default="auto",
                        choices=["auto", "coreml", "pytorch"],
                        help=("Depth inference backend: auto prefers CoreML, "
                              "then PyTorch DA3, then PyTorch DA2"))
    parser.add_argument("--depth-coreml-path", type=str, default=None,
                        help="Path to a converted CoreML depth model package "
                             "(default: <coreml-dir>/<depth-model>.mlpackage)")
    parser.add_argument("--depth-preview-mode", type=str, default="mono",
                        choices=["mono", "alpha", "alpha_color", "overlay"],
                        help=("Preview mode for the right pane: "
                              "mono=grayscale depth, "
                              "alpha=grayscale RGB composited with depth as alpha, "
                              "alpha_color=color RGB composited with depth as alpha, "
                              "overlay=50%% blend of RGB and grayscale depth"))
    parser.add_argument("--ndi-output", type=str, default=None,
                        help="NDI output source name (e.g. 'StreamDiffusion-RGBD'). "
                             "Requires NDI SDK / ndi-python installed.")
    # LoRA stack args
    parser.add_argument("--lora", type=str, action="append", default=[],
                        help="LoRA path or name (can be used multiple times)")
    parser.add_argument("--lora-weight", type=float, action="append", default=[],
                        help="Weight for each --lora (default: 0.8)")
    parser.add_argument("--lora-category", type=str, action="append", default=[],
                        help="Category for each --lora (style/subject/quality)")
    args = parser.parse_args()

    output_width, output_height = _parse_output_size(args.output_size)
    if args.output_width is not None:
        output_width = args.output_width
    if args.output_height is not None:
        output_height = args.output_height

    print("=" * 60)
    print(f"StreamDiffusion for Mac — RGBD Output ({args.model} {args.render_size}x{args.render_size})")
    print(f"  Output resolution: {output_width}x{output_height}")
    print("  CoreML img2img + Depth Anything RGBD generation")
    print("=" * 60)

    prompts = DEFAULT_PROMPTS if args.prompts else None

    # Build LoRA stack from CLI args
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lora_stack = []
    if args.lora:
        weights = args.lora_weight + [0.8] * (len(args.lora) - len(args.lora_weight))
        categories = args.lora_category + ["subject"] * (len(args.lora) - len(args.lora_category))
        for path, weight, category in zip(args.lora, weights, categories):
            resolved = _resolve_lora_path(path, project_dir)
            lora_stack.append({"path": resolved, "weight": weight, "category": category})

    # Use PyTorch pipeline when LoRAs are requested so slider weights affect UNet.
    # SD 1.5 LoRAs are incompatible with SDXS/SD-Turbo UNet attention shapes,
    # so we force the base model to sd-1-5 in LoRA mode.
    use_torch_pipeline = bool(lora_stack)
    if use_torch_pipeline:
        if args.model != "sd-1-5":
            print(f"  [LORA] Model '{args.model}' selected, but SD 1.5 LoRAs require sd-1-5 base model.")
            print(f"  [LORA] Auto-switching to sd-1-5 for this run.")
            args.model = "sd-1-5"
        print("  LoRA stack detected: using PyTorch/MPS pipeline for runtime weight adjustment")
        pipeline = TorchRGBDPipeline(
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
            seed=args.seed,
            lora_stack=lora_stack,
        )
    else:
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

    app = RGBDCameraApp(
        pipeline=pipeline,
        camera_id=args.camera,
        blend_ratio=args.blend,
        ema_alpha=args.ema,
        depth_preview_mode=args.depth_preview_mode,
        ndi_output_name=args.ndi_output,
        output_width=output_width,
        output_height=output_height,
    )
    app.run()


if __name__ == "__main__":
    main()
