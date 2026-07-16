#!/usr/bin/env python3
"""
CoreML Model Conversion Script for StreamDiffusion-Mac

Converts UNet + TinyVAE (encoder/decoder) to CoreML format.
Supported base models: sdxs, sd-turbo, sd-1-5.
SDXS-512 is the optimal model for real-time inference on Apple Silicon,
achieving 22.7 FPS camera img2img on M3 Ultra.

Usage:
    python scripts/convert_models.py
    python scripts/convert_models.py --output-dir ./coreml_models
    python scripts/convert_models.py --model sd-turbo
    python scripts/convert_models.py --model sd-1-5     # SD 1.5 for LoRA
"""
import os
import sys

# Python 3.12 removed distutils; coremltools still imports it.
# Shim via setuptools before importing coremltools.
if "distutils" not in sys.modules:
    import setuptools
    sys.modules["distutils"] = setuptools._distutils

import time
import gc
import argparse
import numpy as np
import torch
import coremltools as ct

# Load centralized model configuration from JSON.
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
from configs import MODEL_CONFIGS

# The conversion script uses a different key name for the CoreML UNet output.
# Build a mapping that preserves the centralized config while adding unet_name.
MODEL_CONFIGS = {
    name: {**cfg, "unet_name": cfg["unet_prefix"]}
    for name, cfg in MODEL_CONFIGS.items()
}


def convert_unet(model_id, hidden_size, save_path, size=512):
    """Convert UNet to CoreML."""
    from diffusers import StableDiffusionPipeline

    print(f"  Loading UNet from {model_id}...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        )
    except ValueError:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
    unet = pipe.unet.eval().float().cpu()

    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.unet(
                sample, timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]

    wrapper = UNetWrapper(unet).eval()

    latent_size = size // 8
    sample = torch.randn(1, 4, latent_size, latent_size)
    timestep = torch.tensor([999.0])
    hidden_states = torch.randn(1, 77, hidden_size)

    print(f"  Tracing UNet at {size}x{size} (latent {latent_size}x{latent_size})...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (sample, timestep, hidden_states))

    print("  Converting to CoreML (this may take several minutes)...")
    t0 = time.time()
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="sample", shape=sample.shape, dtype=np.float16),
            ct.TensorType(name="timestep", shape=timestep.shape, dtype=np.float16),
            ct.TensorType(name="encoder_hidden_states", shape=hidden_states.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="noise_pred", dtype=np.float16),
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    elapsed = time.time() - t0
    print(f"  UNet converted in {elapsed:.1f}s")

    model.save(save_path)
    print(f"  Saved: {save_path}")

    # Return pipe for text encoder conversion
    del unet, wrapper, traced, model
    gc.collect()
    return pipe


def convert_vae_encoder(save_path, size=512):
    """Convert TinyVAE Encoder to CoreML."""
    from diffusers import AutoencoderTiny

    print(f"  Loading TinyVAE...")
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().float().cpu()

    class Wrapper(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.encoder = v.encoder

        def forward(self, x):
            return self.encoder(x)

    wrapper = Wrapper(vae).eval()
    dummy = torch.randn(1, 3, size, size)

    print("  Tracing VAE Encoder...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    print("  Converting to CoreML...")
    t0 = time.time()
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=dummy.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="latent", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    elapsed = time.time() - t0
    print(f"  VAE Encoder converted in {elapsed:.1f}s")

    model.save(save_path)
    print(f"  Saved: {save_path}")

    del vae, wrapper, traced, model
    gc.collect()


def convert_vae_decoder(save_path, size=512):
    """Convert TinyVAE Decoder to CoreML."""
    from diffusers import AutoencoderTiny

    print(f"  Loading TinyVAE...")
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().float().cpu()

    class Wrapper(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.decoder = v.decoder

        def forward(self, x):
            return self.decoder(x)

    wrapper = Wrapper(vae).eval()
    latent_size = size // 8
    dummy = torch.randn(1, 4, latent_size, latent_size)

    print("  Tracing VAE Decoder...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    print("  Converting to CoreML...")
    t0 = time.time()
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latent", shape=dummy.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="image", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    elapsed = time.time() - t0
    print(f"  VAE Decoder converted in {elapsed:.1f}s")

    model.save(save_path)
    print(f"  Saved: {save_path}")

    del vae, wrapper, traced, model
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Convert models to CoreML")
    parser.add_argument("--output-dir", default="coreml_models",
                        help="Output directory for CoreML models")
    parser.add_argument("--model", default="sdxs", choices=list(MODEL_CONFIGS.keys()),
                        help="Model to convert (default: sdxs).")
    parser.add_argument("--size", type=int, default=None,
                        help="Render resolution (default: read from model config or 512).")
    args = parser.parse_args()

    # Resolve output directory relative to project root (two levels up from script).
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    cfg = MODEL_CONFIGS[args.model]
    size = args.size or cfg.get("render_size", 512)
    latent_size = size // 8

    print("=" * 60)
    print(f"CoreML Model Conversion — {args.model} @ {size}x{size}")
    print(f"Output: {os.path.abspath(output_dir)}")
    print("=" * 60)

    # Step 1: UNet
    print(f"\n[1/3] Converting UNet ({cfg['model_id']})...")
    unet_path = os.path.join(output_dir, f"{cfg['unet_name']}.mlpackage")
    pipe = convert_unet(cfg["model_id"], cfg["hidden_size"], unet_path, size=size)

    # Step 2: VAE Encoder
    print(f"\n[2/3] Converting TinyVAE Encoder ({size}x{size})...")
    if size == 512:
        enc_path = os.path.join(output_dir, "taesd_encoder_512.mlpackage")
    else:
        enc_path = os.path.join(output_dir, f"taesd_encoder_{size}.mlpackage")
    convert_vae_encoder(enc_path, size=size)

    # Step 3: VAE Decoder
    print(f"\n[3/3] Converting TinyVAE Decoder ({size}x{size})...")
    if size == 512:
        dec_path = os.path.join(output_dir, "taesd_decoder.mlpackage")
    else:
        dec_path = os.path.join(output_dir, f"taesd_decoder_{size}.mlpackage")
    convert_vae_decoder(dec_path, size=size)

    print("\n" + "=" * 60)
    print("ALL CONVERSIONS COMPLETE!")
    print(f"Models saved to: {os.path.abspath(output_dir)}")
    print("")
    print("Next: Run the camera pipeline:")
    print(f"  python camera.py --prompt 'oil painting style'")
    print("=" * 60)


if __name__ == "__main__":
    main()
