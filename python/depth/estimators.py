# -*- coding: utf-8 -*-
"""
Depth estimation backends for the RGBD pipeline.

Supports:
  - CoreML Depth Anything 3 / V2 (pre-converted .mlpackage)
  - PyTorch Depth Anything V3 (all variants)
  - PyTorch Depth Anything V2 (all Hugging Face variants)
"""

import os
import sys
import types
import numpy as np
import cv2
import torch

from utils.device import default_device
from utils.paths import COREML_DIR


def normalize_depth(depth):
    """Normalize a float depth map to uint8."""
    lo, hi = depth.min(), depth.max()
    if hi - lo < 1e-6:
        return np.zeros_like(depth, dtype=np.uint8)
    norm = (depth - lo) / (hi - lo)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


class CoreMLDepthEstimator:
    """Depth Anything 3 / V2 running as a CoreML model (macOS/Apple Silicon)."""

    INPUT_SIZE = 504
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, model_path):
        import coremltools as ct

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CoreML depth model not found: {model_path}")

        print(f"  Loading CoreML depth model: {model_path}")
        self.model = ct.models.MLModel(
            model_path,
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )

        # Auto-discover input/output tensor names from the model spec.
        spec = self.model.get_spec()
        in_feat = spec.description.input[0]
        out_feat = spec.description.output[0]
        self.in_name = in_feat.name
        self.out_name = out_feat.name
        in_shape = list(in_feat.type.multiArrayType.shape)
        out_shape = list(out_feat.type.multiArrayType.shape)
        self._input_rank = len(in_shape)
        print(f"    Input : {self.in_name} -> {in_shape} (rank {self._input_rank})")
        print(f"    Output: {self.out_name} -> {out_shape}")

    def estimate(self, rgb_np):
        """Estimate depth for a single RGB uint8 image."""
        h, w = rgb_np.shape[:2]

        # 1. Resize preserving aspect ratio and pad to INPUT_SIZE.
        scale = min(self.INPUT_SIZE / w, self.INPUT_SIZE / h)
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        resized = cv2.resize(rgb_np, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype=np.float32)
        padded[:scaled_h, :scaled_w] = resized.astype(np.float32) / 255.0

        # 2. ImageNet normalization.
        normalized = (padded - self.IMAGENET_MEAN) / self.IMAGENET_STD

        # 3. Add batch dims expected by the model (DA3: 5D, DA2: 4D).
        transposed = normalized.transpose(2, 0, 1)
        if self._input_rank == 5:
            x = np.ascontiguousarray(transposed[None, None])
        else:
            x = np.ascontiguousarray(transposed[None])

        # 4. CoreML inference.
        pred = self.model.predict({self.in_name: x})
        depth = pred[self.out_name]

        # 5. Remove batch/channel dims, crop away padding, and resize back.
        depth = np.squeeze(depth)
        if depth.ndim != 2:
            raise RuntimeError(f"Unexpected CoreML depth output shape after squeeze: {depth.shape}")

        depth_cropped = depth[:scaled_h, :scaled_w]
        depth_full = cv2.resize(depth_cropped.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return normalize_depth(depth_full)


class TorchDepthEstimator:
    """Depth Anything V3/V2 via PyTorch / Transformers."""

    # Mapping from our short names to HuggingFace repo ids.
    DA3_NAMES = {
        "da3-small": "depth-anything/DA3-SMALL",
        "da3-base": "depth-anything/DA3-BASE",
        "da3-large": "depth-anything/DA3-LARGE",
    }
    DA2_NAMES = {
        "da2-small": "depth-anything/Depth-Anything-V2-Small-hf",
        "da2-base": "depth-anything/Depth-Anything-V2-Base-hf",
        "da2-large": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    def __init__(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self._pipe = None
        self._da3_model = None

        if model_name.startswith("da3-"):
            self._load_da3()
        elif model_name.startswith("da2-"):
            self._load_da2()
        else:
            raise ValueError(f"Unknown PyTorch depth model: {model_name}")

    @staticmethod
    def _da3_available():
        # Insert depth-anything-3 source path and stub heavy deps before import.
        DA3_SRC = "/tmp/depth-anything-3/src"
        if os.path.isdir(DA3_SRC) and DA3_SRC not in sys.path:
            sys.path.insert(0, DA3_SRC)
        for _mod_name in [
            "open3d", "trimesh", "e3nn", "pycolmap", "pillow_heif",
            "fastapi", "uvicorn", "typer", "requests", "gsplat",
        ]:
            if _mod_name not in sys.modules:
                sys.modules[_mod_name] = types.ModuleType(_mod_name)
        try:
            from depth_anything_3.api import DepthAnything3  # noqa: F401
            return True
        except Exception:
            return False

    def _load_da3(self):
        repo_id = self.DA3_NAMES.get(self.model_name)
        if repo_id is None:
            raise ValueError(f"Unknown DA3 model: {self.model_name}")

        # Same path+stub setup as _da3_available in case init order differs.
        DA3_SRC = "/tmp/depth-anything-3/src"
        if os.path.isdir(DA3_SRC) and DA3_SRC not in sys.path:
            sys.path.insert(0, DA3_SRC)
        for _mod_name in [
            "open3d", "trimesh", "e3nn", "pycolmap", "pillow_heif",
            "fastapi", "uvicorn", "typer", "requests", "gsplat",
        ]:
            if _mod_name not in sys.modules:
                sys.modules[_mod_name] = types.ModuleType(_mod_name)
        from depth_anything_3.api import DepthAnything3
        # MPS has autocast dtype compatibility issues with DA3's scaled_dot_product_attention
        # and upsample_bicubic2d is not implemented on MPS. Fallback to CPU for DA3.
        device = self.device
        if device == "mps":
            print("  [DA3] MPS device has compatibility issues; using CPU for depth estimation.")
            device = "cpu"
        self._da3_model = DepthAnything3.from_pretrained(repo_id).to(device)
        self._da3_model.eval()

    def _load_da2(self):
        repo_id = self.DA2_NAMES.get(self.model_name)
        if repo_id is None:
            raise ValueError(f"Unknown DA2 model: {self.model_name}")
        from transformers import pipeline
        self._pipe = pipeline(
            "depth-estimation",
            model=repo_id,
            device=self.device,
        )

    def estimate(self, rgb_np):
        """Estimate depth for a single RGB uint8 image."""
        h, w = rgb_np.shape[:2]
        if self.model_name.startswith("da3-"):
            depth = self._estimate_da3(rgb_np)
        else:
            depth = self._estimate_da2(rgb_np)

        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth

    def _estimate_da3(self, rgb_np):
        with torch.no_grad():
            prediction = self._da3_model.inference([rgb_np])
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        return normalize_depth(depth)

    def _estimate_da2(self, rgb_np):
        from PIL import Image
        pred = self._pipe(Image.fromarray(rgb_np))
        depth = np.asarray(pred["predicted_depth"], dtype=np.float32)
        return normalize_depth(depth)


class DepthEstimator:
    """Picks the best available depth estimator for the current platform."""

    # Model family used when auto-selecting CoreML.
    _DA3_MODELS = ("da3-small", "da3-base", "da3-large")
    _DA2_MODELS = ("da2-small", "da2-base", "da2-large")

    def __init__(self, model_name="auto", backend="auto", coreml_path=None, device=None):
        self.device = device or default_device()

        # Resolve CoreML model path based on requested model.
        if coreml_path is None:
            if model_name in self._DA2_MODELS:
                coreml_path = os.path.join(COREML_DIR, f"{model_name.replace('-', '_')}.mlpackage")
            elif model_name in self._DA3_MODELS:
                coreml_path = os.path.join(COREML_DIR, f"{model_name.replace('-', '_')}.mlpackage")
            else:
                # auto or unknown -> default to da3_small
                coreml_path = os.path.join(COREML_DIR, "da3_small.mlpackage")

        # Auto-select backend.
        # Prefer CoreML small variants when the requested base/large CoreML model
        # is missing, because PyTorch base/large models (especially DA3 on CPU)
        # are much slower than the CoreML small variant.
        if backend == "auto":
            if os.path.exists(coreml_path):
                backend = "coreml"
            else:
                fallback_coreml = None
                if model_name in self._DA3_MODELS:
                    candidate = os.path.join(COREML_DIR, "da3_small.mlpackage")
                    if os.path.exists(candidate):
                        fallback_coreml = ("da3-small", candidate)
                if fallback_coreml is None and model_name in self._DA2_MODELS:
                    candidate = os.path.join(COREML_DIR, "da2_small.mlpackage")
                    if os.path.exists(candidate):
                        fallback_coreml = ("da2-small", candidate)

                if fallback_coreml is not None:
                    fallback_name, fallback_path = fallback_coreml
                    print(f"  [WARN] CoreML model for '{model_name}' not found at {coreml_path}.")
                    print(f"  [WARN] Falling back to CoreML '{fallback_name}' for fast depth estimation.")
                    print(f"         Run the conversion script to use '{model_name}' with CoreML:")
                    print(f"         python scripts/convert_da{3 if model_name in self._DA3_MODELS else 2}_coreml.py --variant {model_name.split('-')[1]}")
                    backend = "coreml"
                    model_name = fallback_name
                    coreml_path = fallback_path
                elif model_name in ("auto",) + self._DA3_MODELS and TorchDepthEstimator._da3_available():
                    backend = "pytorch"
                    if model_name == "auto":
                        model_name = "da3-small"
                else:
                    backend = "pytorch"
                    if model_name == "auto":
                        print("  Depth Anything V3/CoreML not found; using Depth Anything V2 Small.")
                        model_name = "da2-small"

        self.backend = backend
        self.model_name = model_name

        if backend == "coreml":
            self._estimator = CoreMLDepthEstimator(coreml_path)
        elif backend == "pytorch":
            self._estimator = TorchDepthEstimator(model_name, self.device)
        else:
            raise ValueError(f"Unknown depth backend: {backend}")

        print(f"  Depth estimator ready: {backend} ({model_name}) on {self.device}")

    def estimate(self, rgb_np):
        return self._estimator.estimate(rgb_np)
