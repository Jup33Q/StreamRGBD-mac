#!/usr/bin/env python3
"""
StreamDiffusion for Mac — LoRA-enhanced Camera Pipeline
Supports loading SD 1.5 compatible LoRAs for real-time style transfer.

Usage:
    # Basic (no LoRA)
    python camera_lora.py --prompt "oil painting style"

    # With LoRA
    python camera_lora.py --prompt "oil painting style" --lora watercolor --lora-weight 0.8

    # Multiple LoRAs (stacked)
    python camera_lora.py --prompt "cyberpunk neon" \
        --lora detail_tweaker --lora-weight 0.5 \
        --lora epi_noiseoffset --lora-weight 0.7

    # Custom LoRA path
    python camera_lora.py --prompt "anime style" --lora /path/to/custom.safetensors

    # List available LoRAs
    python camera_lora.py --list-loras

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipelines.coreml import Pipeline, COREML_DIR
from pipelines.lora import (
    LoRAEnhancedPipeline,
    resolve_lora_paths,
    list_local_loras,
    LORAS_DIR,
)
from apps.camera import CameraApp
from configs import MODEL_CONFIGS, DEFAULT_PROMPTS

# Backward-compatible re-exports.
__all__ = [
    "Pipeline",
    "COREML_DIR",
    "LoRAEnhancedPipeline",
    "CameraApp",
    "resolve_lora_paths",
    "list_local_loras",
    "LORAS_DIR",
]


def main():
    parser = argparse.ArgumentParser(description="StreamDiffusion for Mac — LoRA-enhanced Camera")
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
    # LoRA args
    parser.add_argument("--lora", type=str, action="append", default=[],
                        help="LoRA name or path (can be used multiple times)")
    parser.add_argument("--lora-weight", type=float, action="append", default=[],
                        help="Weight for each LoRA (default: 0.8)")
    parser.add_argument("--lora-dir", type=str, default=LORAS_DIR,
                        help="Directory containing LoRA .safetensors files")
    parser.add_argument("--list-loras", action="store_true", help="List available local LoRAs")
    parser.add_argument("--coreml-dir", type=str, default=COREML_DIR)
    args = parser.parse_args()

    if args.list_loras:
        list_local_loras(args.lora_dir)
        return

    print("=" * 60)
    print(f"StreamDiffusion for Mac — LoRA ({args.model} {args.render_size}x{args.render_size})")
    print("  CoreML-accelerated real-time camera with LoRA styles")
    print("=" * 60)

    # Resolve LoRA paths
    lora_paths = []
    lora_weights = []
    if args.lora:
        lora_paths = resolve_lora_paths(args.lora, args.lora_dir)
        if not lora_paths:
            print("No valid LoRAs found. Run with --list-loras to check.")
            print(f"LoRA directory: {args.lora_dir}")
        # Pad weights
        lora_weights = args.lora_weight + [0.8] * (len(lora_paths) - len(args.lora_weight))
        lora_weights = lora_weights[:len(lora_paths)]

    prompts = DEFAULT_PROMPTS if args.prompts else None

    pipeline = LoRAEnhancedPipeline(
        model_name=args.model,
        render_size=args.render_size,
        output_size=args.output_size,
        prompt=args.prompt,
        strength=args.strength,
        prompts=prompts,
        latent_feedback=args.feedback,
        coreml_dir=args.coreml_dir,
        lora_paths=lora_paths,
        lora_weights=lora_weights,
    )

    app = CameraApp(
        pipeline=pipeline,
        camera_id=0,
        blend_ratio=args.blend,
        ema_alpha=args.ema,
    )
    app.run()


if __name__ == "__main__":
    main()
