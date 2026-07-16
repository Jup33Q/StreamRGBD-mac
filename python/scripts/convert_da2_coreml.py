#!/usr/bin/env python3
"""
Convert Depth Anything V2 variants (small/base/large) to CoreML .mlpackage.

The transformers DA2 model uses bicubic interpolation which coremltools does not
implement, so we temporarily patch torch.nn.functional.interpolate to bilinear
mode during tracing. This produces a slightly different but functionally
equivalent depth model.

Usage:
    python scripts/convert_da2_coreml.py --variant small
    python scripts/convert_da2_coreml.py --variant base
    python scripts/convert_da2_coreml.py --variant large --output-dir ./coreml_models
"""
import os
import sys
import time
import argparse
import gc

# Python 3.12 removed distutils; coremltools still imports it.
if "distutils" not in sys.modules:
    import setuptools
    sys.modules["distutils"] = setuptools._distutils

import torch
import torch.nn.functional as F
import coremltools as ct
import numpy as np
from transformers import pipeline


REPO_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def main():
    parser = argparse.ArgumentParser(description="Convert DA2 variants to CoreML")
    parser.add_argument("--variant", choices=["small", "base", "large"], default="small",
                        help="DA2 variant to convert (default: small)")
    parser.add_argument("--output-dir", default="coreml_models",
                        help="Output directory for CoreML model")
    parser.add_argument("--size", type=int, default=504,
                        help="Input resolution (default: 504)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    repo_id = REPO_IDS[args.variant]
    output_name = f"da2_{args.variant}.mlpackage"
    save_path = os.path.join(output_dir, output_name)

    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}, skipping conversion.")
        return

    print("=" * 60)
    print(f"CoreML Conversion — Depth Anything V2 {args.variant.upper()}")
    print(f"Repo: {repo_id}")
    print(f"Output: {os.path.abspath(save_path)}")
    print("=" * 60)

    # Patch bicubic interpolate to bilinear because coremltools does not
    # implement upsample_bicubic2d.
    _original_interpolate = F.interpolate
    def _patched_interpolate(input, size=None, scale_factor=None, mode='nearest',
                             align_corners=None, *args, **kwargs):
        if mode == 'bicubic':
            mode = 'bilinear'
        return _original_interpolate(input, size, scale_factor, mode,
                                     align_corners, *args, **kwargs)
    F.interpolate = _patched_interpolate

    try:
        print(f"\n[1/2] Loading DA2-{args.variant.upper()} from HuggingFace...")
        pipe = pipeline(
            "depth-estimation",
            model=repo_id,
            device=-1,
        )
        model = pipe.model.eval().float().cpu()

        class Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                out = self.m(pixel_values=x)
                return out.predicted_depth

        w = Wrapper(model)
        dummy = torch.randn(1, 3, args.size, args.size)

        print("[2/2] Tracing and converting to CoreML...")
        with torch.no_grad():
            traced = torch.jit.trace(w, dummy)

        t0 = time.time()
        m = ct.convert(
            traced,
            inputs=[ct.TensorType(name="image", shape=dummy.shape, dtype=np.float32)],
            outputs=[ct.TensorType(name="depth", dtype=np.float32)],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS14,
            convert_to="mlprogram",
        )
        print(f"  Converted in {time.time() - t0:.1f}s")

        m.save(save_path)
        print(f"  Saved: {save_path}")

        del m, traced, w, model, pipe
        gc.collect()

    finally:
        F.interpolate = _original_interpolate

    print("\n" + "=" * 60)
    print(f"DA2-{args.variant.upper()} CoreML conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
