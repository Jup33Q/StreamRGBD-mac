# -*- coding: utf-8 -*-
"""Device selection helpers."""

import torch


def default_device():
    """Pick MPS on Apple Silicon, CUDA if present, otherwise CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Backward-compatible alias used by existing code.
_default_device = default_device
