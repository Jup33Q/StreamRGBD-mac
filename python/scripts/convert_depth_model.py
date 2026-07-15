#!/usr/bin/env python3
"""
Convert Depth-Anything-V3 Small to CoreML .mlpackage.

This script loads the DA3-SMALL model from HuggingFace, traces the depth
prediction path, and converts it to CoreML format for Apple Silicon.

Usage:
    python scripts/convert_depth_model.py
    python scripts/convert_depth_model.py --output-dir ./coreml_models
"""
import os
import sys
import time
import argparse
import gc
import types

# Python 3.12 removed distutils; coremltools still imports it.
# Shim via setuptools before importing coremltools.
if "distutils" not in sys.modules:
    import setuptools
    sys.modules["distutils"] = setuptools._distutils

import numpy as np
import torch
import coremltools as ct

# Add the depth-anything-3 source to the path (cloned without installing).
DEPTH_ANYTHING_3_SRC = "/tmp/depth-anything-3/src"
if os.path.exists(DEPTH_ANYTHING_3_SRC):
    sys.path.insert(0, DEPTH_ANYTHING_3_SRC)

# Stub out heavy dependencies that are imported by depth_anything_3 but not
# needed for CoreML conversion (export/GS/video rendering modules).
for _mod_name in [
    "open3d",
    "trimesh",
    "e3nn",
    "pycolmap",
    "pillow_heif",
    "fastapi",
    "uvicorn",
    "typer",
    "requests",
    "gsplat",
]:
    sys.modules[_mod_name] = types.ModuleType(_mod_name)

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.dinov2.layers.rope import PositionGetter, RotaryPositionEmbedding2D


# Precompute position grid for 504x504 (36x36 patches, 1296 positions, 2 coords).
# This avoids torch.cartesian_prod which is not supported by coremltools.
_PRECOMPUTED_POSITIONS = {}

def _patched_position_getter(self, batch_size, height, width, device):
    key = (batch_size, height, width, str(device))
    if key not in _PRECOMPUTED_POSITIONS:
        y_coords = torch.arange(height, device=device)
        x_coords = torch.arange(width, device=device)
        # Use meshgrid + stack instead of cartesian_prod
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        positions = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
        _PRECOMPUTED_POSITIONS[key] = positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()
    return _PRECOMPUTED_POSITIONS[key].to(device)


# Patch PositionGetter to avoid cartesian_prod.
PositionGetter.__call__ = _patched_position_getter


# Precompute max_position for 504x504 (35 max coordinate).
def _patched_rope_forward(self, tokens, positions):
    assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
    assert (
        positions.ndim == 3 and positions.shape[-1] == 2
    ), "Positions must have shape (batch_size, n_tokens, 2)"

    feature_dim = tokens.size(-1) // 2
    # Hardcode max_position for 36x36 grid (504/14 = 36) + 1 for cls_token shift.
    max_position = 37

    cos_comp, sin_comp = self._compute_frequency_components(
        feature_dim, max_position, tokens.device, tokens.dtype
    )

    vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
    vertical_features = self._apply_1d_rope(
        vertical_features, positions[..., 0], cos_comp, sin_comp
    )
    horizontal_features = self._apply_1d_rope(
        horizontal_features, positions[..., 1], cos_comp, sin_comp
    )
    return torch.cat((vertical_features, horizontal_features), dim=-1)


RotaryPositionEmbedding2D.forward = _patched_rope_forward


def resolve_output_dir(args_output_dir):
    """Resolve output directory relative to project root."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, args_output_dir)


class DA3DepthWrapper(torch.nn.Module):
    """Wrapper that extracts only the depth tensor from DA3 output."""

    def __init__(self, da3_model):
        super().__init__()
        self.backbone = da3_model.model

    def forward(self, image):
        output = self.backbone(
            image,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=[],
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy="saddle_balanced",
        )
        return output.depth


def convert_da3_small(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "da3_small.mlpackage")

    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}, skipping conversion.")
        return save_path

    print("=" * 60)
    print("CoreML DA3-Small Conversion")
    print(f"Output: {save_path}")
    print("=" * 60)

    # Use CPU for conversion. MPS lacks some ops and the resulting CoreML model
    # will run on Apple Silicon via CoreML anyway.
    device = "cpu"
    print(f"\n[1/4] Loading DA3-SMALL from HuggingFace on {device}...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
    model = model.to(device).eval()

    print("\n[2/4] Creating wrapper and running dry forward...")
    wrapper = DA3DepthWrapper(model).eval().to(device)

    dummy_input = torch.randn(1, 1, 3, 504, 504, device=device)
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
                shape=(1, 1, 3, 504, 504),
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
    parser = argparse.ArgumentParser(description="Convert DA3-Small to CoreML")
    parser.add_argument(
        "--output-dir",
        default="coreml_models",
        help="Output directory for CoreML models (relative to project root)",
    )
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.output_dir)
    convert_da3_small(output_dir)

    print("\n" + "=" * 60)
    print("DA3-SMALL CoreML conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
