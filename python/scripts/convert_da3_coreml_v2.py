#!/usr/bin/env python3
"""
Convert DA3-SMALL to CoreML .mlpackage (v2).

Uses direct network construction + safetensors weight loading
(inspired by LSQzzx/Depth-Anything-3-for-CoreML) to avoid the int()
conversion issue with torch.jit.trace on the full from_pretrained model.

Usage:
    python scripts/convert_da3_coreml_v2.py
    python scripts/convert_da3_coreml_v2.py --output-dir ./coreml_models
"""
import os
import sys
import time
import argparse
import gc
import types

# Python 3.12 removed distutils; coremltools still imports it.
if "distutils" not in sys.modules:
    import setuptools
    sys.modules["distutils"] = setuptools._distutils

import numpy as np
import torch
import coremltools as ct
from safetensors.torch import load_file

# Add the depth-anything-3 source to the path.
DEPTH_ANYTHING_3_SRC = "/tmp/depth-anything-3/src"
if os.path.exists(DEPTH_ANYTHING_3_SRC):
    sys.path.insert(0, DEPTH_ANYTHING_3_SRC)

# Stub out heavy dependencies.
for _mod_name in [
    "open3d", "trimesh", "e3nn", "pycolmap", "pillow_heif",
    "fastapi", "uvicorn", "typer", "requests", "gsplat",
]:
    sys.modules[_mod_name] = types.ModuleType(_mod_name)

from depth_anything_3.model.da3 import DepthAnything3Net
from depth_anything_3.model.dinov2.dinov2 import DinoV2
from depth_anything_3.model.dualdpt import DualDPT

INPUT_SIZE = 504
DEFAULT_COREML_OUTPUT = "da3_small.mlpackage"


def _build_backbone() -> DinoV2:
    return DinoV2(
        name="vits",
        out_layers=[5, 7, 9, 11],
        alt_start=4,
        qknorm_start=4,
        rope_start=4,
        cat_token=True,
    )


def _build_head() -> DualDPT:
    return DualDPT(
        dim_in=768,
        output_dim=2,
        features=64,
        out_channels=[48, 96, 192, 384],
        pos_embed=False,
    )


def build_depth_estimator() -> DepthAnything3Net:
    return DepthAnything3Net(net=_build_backbone(), head=_build_head())


def load_weights(model: DepthAnything3Net, weights_path: str) -> None:
    checkpoint = load_file(weights_path)
    sanitized = {k.replace("model.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(sanitized, strict=False)


class DA3DepthCoreMLWrapper(torch.nn.Module):
    """Wrapper that returns only the depth channel (first output channel) for CoreML."""

    def __init__(self, depth_model: DepthAnything3Net):
        super().__init__()
        self.depth_model = depth_model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.depth_model(image)
        # output.depth has shape (B, H, W); add a channel dim so squeeze in CoreMLDepthEstimator works
        depth = output.depth[:, None, :, :]  # (B, 1, H, W)
        return depth


def convert_da3_small(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, DEFAULT_COREML_OUTPUT)

    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}, skipping conversion.")
        return save_path

    print("=" * 60)
    print("CoreML DA3-SMALL Conversion (v2)")
    print(f"Output: {save_path}")
    print("=" * 60)

    # Resolve weight path from HuggingFace cache
    weight_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--depth-anything--DA3-SMALL/"
        "snapshots/e08cab65ca0ec38e7826075418411ab90cab4da3/model.safetensors"
    )
    if not os.path.exists(weight_path):
        print(f"[ERROR] Weight file not found at {weight_path}")
        print("  Please ensure DA3-SMALL is downloaded from HuggingFace.")
        sys.exit(1)

    device = "cpu"
    print(f"\n[1/4] Building network and loading weights on {device}...")
    model = build_depth_estimator()
    load_weights(model, weight_path)
    model = model.to(device).eval()

    print("\n[2/4] Creating CoreML wrapper and running dry forward...")
    wrapper = DA3DepthCoreMLWrapper(model).eval().to(device)
    dummy_input = torch.randn(1, 1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    with torch.no_grad():
        test_out = wrapper(dummy_input)
    print(f"    Dry forward output shape: {test_out.shape}")
    print(f"    Output dtype: {test_out.dtype}")

    print("\n[3/4] Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_input)

    print("\n[4/4] Converting to CoreML...")
    t0 = time.time()
    coreml_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="image",
                shape=(1, 1, 3, INPUT_SIZE, INPUT_SIZE),
                dtype=np.float32,
            ),
        ],
        outputs=[
            ct.TensorType(name="depth", dtype=np.float32),
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    elapsed = time.time() - t0
    print(f"    Conversion complete in {elapsed:.1f}s")

    coreml_model.save(save_path)
    print(f"    Saved: {save_path}")

    del traced, coreml_model, wrapper, model
    gc.collect()

    return save_path


def main():
    parser = argparse.ArgumentParser(description="Convert DA3-Small to CoreML (v2)")
    parser.add_argument(
        "--output-dir",
        default="coreml_models",
        help="Output directory for CoreML models (relative to project root)",
    )
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", args.output_dir)
    output_dir = os.path.abspath(output_dir)
    convert_da3_small(output_dir)

    print("\n" + "=" * 60)
    print("DA3-SMALL CoreML conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
