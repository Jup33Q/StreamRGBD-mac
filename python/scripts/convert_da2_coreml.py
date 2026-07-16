#!/usr/bin/env python3
"""
Convert Depth Anything V2 Small to CoreML .mlpackage.

The transformers DA2 model uses bicubic interpolation which coremltools does not
implement, so we temporarily patch torch.nn.functional.interpolate to bilinear
mode during tracing. This produces a slightly different but functionally
equivalent depth model.

Usage:
    python scripts/convert_da2_coreml.py
    python scripts/convert_da2_coreml.py --output-dir ./coreml_models
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


def main():
    parser = argparse.ArgumentParser(description="Convert DA2-Small to CoreML")
    parser.add_argument("--output-dir", default="coreml_models",
                        help="Output directory for CoreML model")
    parser.add_argument("--size", type=int, default=504,
                        help="Input resolution (default: 504)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("CoreML Conversion — Depth Anything V2 Small")
    print(f"Output: {os.path.abspath(output_dir)}")
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
        print("\n[1/2] Loading DA2-Small from HuggingFace...")
        pipe = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
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

        save_path = os.path.join(output_dir, "da2_small.mlpackage")
        m.save(save_path)
        print(f"  Saved: {save_path}")

        del m, traced, w, model, pipe
        gc.collect()

    finally:
        F.interpolate = _original_interpolate

    print("\n" + "=" * 60)
    print("DA2-Small CoreML conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
