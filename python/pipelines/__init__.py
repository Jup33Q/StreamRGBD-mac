#!/usr/bin/env python3
"""StreamDiffusion image generation pipelines."""
from pipelines.coreml import (
    COREML_DIR,
    Pipeline,
    ensure_vae_encoder,
    ensure_vae_decoder,
)
from pipelines.rgbd import RGBDPipeline
from pipelines.torch import TorchPipeline, TorchRGBDPipeline
from pipelines.lora import (
    LoRAEnhancedPipeline,
    resolve_lora_paths,
    list_local_loras,
)

__all__ = [
    "COREML_DIR",
    "Pipeline",
    "ensure_vae_encoder",
    "ensure_vae_decoder",
    "RGBDPipeline",
    "TorchPipeline",
    "TorchRGBDPipeline",
    "LoRAEnhancedPipeline",
    "resolve_lora_paths",
    "list_local_loras",
]
