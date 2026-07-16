#!/usr/bin/env python3
"""
Convert Depth Anything V3 variants (small/base/large) to CoreML .mlpackage.

Uses direct network construction + safetensors weight loading.
Includes monkey-patches for torch.jit.trace compatibility with coremltools.

Usage:
    python scripts/convert_da3_coreml.py --variant small
    python scripts/convert_da3_coreml.py --variant base
    python scripts/convert_da3_coreml.py --variant large --output-dir ./coreml_models
"""
import os
import sys
import time
import argparse
import gc
import types
import math

# Python 3.12 removed distutils; coremltools still imports it.
if "distutils" not in sys.modules:
    import setuptools
    sys.modules["distutils"] = setuptools._distutils

import numpy as np
import torch
import coremltools as ct
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

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
from depth_anything_3.model.dinov2.layers.rope import PositionGetter, RotaryPositionEmbedding2D
from depth_anything_3.model.dinov2 import vision_transformer

# ------------------------------------------------------------------
# Monkey-patches for torch.jit.trace / coremltools compatibility
# ------------------------------------------------------------------

# Patch 1: PositionGetter - avoid cartesian_prod (not supported by coremltools)
_PRECOMPUTED_POSITIONS = {}


def _patched_position_getter(self, batch_size, height, width, device):
    key = (batch_size, height, width, str(device))
    if key not in _PRECOMPUTED_POSITIONS:
        y_coords = torch.arange(height, device=device)
        x_coords = torch.arange(width, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        positions = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
        _PRECOMPUTED_POSITIONS[key] = positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()
    return _PRECOMPUTED_POSITIONS[key].to(device)


PositionGetter.__call__ = _patched_position_getter


# Patch 2: RotaryPositionEmbedding2D - avoid int(tensor) which coremltools can't handle
# Hardcode max_position for 504x504 (36x36 grid + 1 for cls_token shift = 37)
def _patched_rope_forward(self, tokens, positions):
    assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
    assert (
        positions.ndim == 3 and positions.shape[-1] == 2
    ), "Positions must have shape (batch_size, n_tokens, 2)"

    feature_dim = tokens.size(-1) // 2
    max_position = 37  # 504/14 = 36 patches per side + cls_token offset

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


# Patch 3: interpolate_pos_encoding - avoid int(math.sqrt(N))
# math.isqrt() returns int directly without needing int()
_original_interpolate = vision_transformer.DinoVisionTransformer.interpolate_pos_encoding


def _patched_interpolate_pos_encoding(self, x, w, h):
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return self.pos_embed
    pos_embed = self.pos_embed.float()
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // self.patch_size
    h0 = h // self.patch_size
    M = math.isqrt(N)  # Use math.isqrt() instead of int(math.sqrt())
    assert N == M * M
    kwargs = {}
    if self.interpolate_offset:
        sx = float(w0 + self.interpolate_offset) / M
        sy = float(h0 + self.interpolate_offset) / M
        kwargs["scale_factor"] = (sx, sy)
    else:
        kwargs["size"] = (w0, h0)
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
        mode="bicubic",
        antialias=self.interpolate_antialias,
        **kwargs,
    )
    assert (w0, h0) == patch_pos_embed.shape[-2:]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)


vision_transformer.DinoVisionTransformer.interpolate_pos_encoding = _patched_interpolate_pos_encoding


# ------------------------------------------------------------------
# Model construction and conversion
# ------------------------------------------------------------------

INPUT_SIZE = 504

# Per-variant architecture specs matching DA3 yaml configs.
VARIANT_SPECS = {
    "small": {
        "repo_id": "depth-anything/DA3-SMALL",
        "backbone_name": "vits",
        "out_layers": [5, 7, 9, 11],
        "alt_start": 4,
        "qknorm_start": 4,
        "rope_start": 4,
        "cat_token": True,
        "head_dim_in": 768,
        "head_features": 64,
        "head_out_channels": [48, 96, 192, 384],
    },
    "base": {
        "repo_id": "depth-anything/DA3-BASE",
        "backbone_name": "vitb",
        "out_layers": [5, 7, 9, 11],
        "alt_start": 4,
        "qknorm_start": 4,
        "rope_start": 4,
        "cat_token": True,
        "head_dim_in": 1536,
        "head_features": 128,
        "head_out_channels": [96, 192, 384, 768],
    },
    "large": {
        "repo_id": "depth-anything/DA3-LARGE",
        "backbone_name": "vitl",
        "out_layers": [11, 15, 19, 23],
        "alt_start": 8,
        "qknorm_start": 8,
        "rope_start": 8,
        "cat_token": True,
        "head_dim_in": 2048,
        "head_features": 256,
        "head_out_channels": [256, 512, 1024, 1024],
    },
}


def _build_backbone(spec) -> DinoV2:
    return DinoV2(
        name=spec["backbone_name"],
        out_layers=spec["out_layers"],
        alt_start=spec["alt_start"],
        qknorm_start=spec["qknorm_start"],
        rope_start=spec["rope_start"],
        cat_token=spec["cat_token"],
    )


def _build_head(spec) -> DualDPT:
    return DualDPT(
        dim_in=spec["head_dim_in"],
        output_dim=2,
        features=spec["head_features"],
        out_channels=spec["head_out_channels"],
        pos_embed=False,
    )


def build_depth_estimator(spec) -> DepthAnything3Net:
    return DepthAnything3Net(net=_build_backbone(spec), head=_build_head(spec))


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


def _download_weights(repo_id: str, local_dir: str = None) -> str:
    """Download model.safetensors from HuggingFace and return the local path."""
    print(f"  Downloading weights from {repo_id} ...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"  Weights cached at: {path}")
    return path


def convert_da3_variant(variant: str, output_dir: str) -> str:
    spec = VARIANT_SPECS.get(variant)
    if spec is None:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(VARIANT_SPECS.keys())}")

    os.makedirs(output_dir, exist_ok=True)
    output_name = f"da3_{variant}.mlpackage"
    save_path = os.path.join(output_dir, output_name)

    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}, skipping conversion.")
        return save_path

    print("=" * 60)
    print(f"CoreML DA3-{variant.upper()} Conversion")
    print(f"Repo: {spec['repo_id']}")
    print(f"Output: {save_path}")
    print("=" * 60)

    weight_path = _download_weights(spec["repo_id"])
    if not os.path.exists(weight_path):
        print(f"[ERROR] Weight file not found at {weight_path}")
        sys.exit(1)

    device = "cpu"
    print(f"\n[1/4] Building network and loading weights on {device}...")
    model = build_depth_estimator(spec)
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
    parser = argparse.ArgumentParser(description="Convert DA3 variants to CoreML")
    parser.add_argument(
        "--variant",
        choices=["small", "base", "large"],
        default="small",
        help="DA3 variant to convert (default: small)",
    )
    parser.add_argument(
        "--output-dir",
        default="coreml_models",
        help="Output directory for CoreML models (relative to project root)",
    )
    args = parser.parse_args()

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", args.output_dir
    )
    output_dir = os.path.abspath(output_dir)
    convert_da3_variant(args.variant, output_dir)

    print("\n" + "=" * 60)
    print(f"DA3-{args.variant.upper()} CoreML conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
