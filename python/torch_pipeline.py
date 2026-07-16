#!/usr/bin/env python3
"""
StreamDiffusion for Mac — PyTorch/MPS Pipeline with Runtime LoRA Adapters

A thin re-export of the LoRA-capable PyTorch pipeline package.

Usage:
    from torch_pipeline import TorchPipeline, TorchRGBDPipeline
    pipeline = TorchPipeline(
        model_name="sdxs",
        render_size=512,
        output_size=512,
        prompt="oil painting style",
        lora_stack=[{"path": "loras/pixelart_redmond.safetensors", "weight": 0.8, "category": "style"}],
    )
    pipeline.set_lora_stack([...])  # runtime update

Note:
    This is slower than the CoreML Pipeline but is required when LoRA weights
    must be adjusted during inference (e.g. via GUI sliders).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipelines.torch import TorchPipeline, TorchRGBDPipeline

__all__ = ["TorchPipeline", "TorchRGBDPipeline"]
