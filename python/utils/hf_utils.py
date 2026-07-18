# -*- coding: utf-8 -*-
"""HuggingFace helpers for offline-first model loading."""
import os
import socket
import urllib.parse


def _hf_endpoint_reachable(timeout=3.0):
    """Return True if the configured HF endpoint accepts TCP connections."""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    host = urllib.parse.urlparse(endpoint).hostname
    if not host:
        return False
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except OSError:
        return False


def from_pretrained_local_first(load_fn, model_id, **kwargs):
    """Load a HuggingFace model, preferring the local cache.

    If the model is fully cached locally, load it without any network access.
    Otherwise fall back to an online download unless ``HF_HUB_OFFLINE=1`` is
    set in the environment or the HF endpoint is unreachable (in which case
    the original local error is re-raised immediately instead of hanging on
    network timeouts).

    Args:
        load_fn: Callable that accepts the model id and kwargs, e.g.
            ``StableDiffusionPipeline.from_pretrained``.
        model_id: HuggingFace repo id (e.g. ``"IDKiro/sdxs-512-0.9"``).
        **kwargs: Extra arguments forwarded to ``load_fn``.

    Returns:
        The loaded model/pipeline object.
    """
    try:
        return load_fn(model_id, local_files_only=True, **kwargs)
    except Exception as e:
        if os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes"):
            raise RuntimeError(
                f"Model {model_id!r} not found in local cache and "
                "HF_HUB_OFFLINE=1 disables downloads."
            ) from e
        if not _hf_endpoint_reachable():
            raise RuntimeError(
                f"Model {model_id!r} is not fully cached locally and the "
                f"HuggingFace endpoint is unreachable. Check your network, or "
                f"set HF_ENDPOINT=https://hf-mirror.com to use the mirror. "
                f"(local load error: {e})"
            ) from e
        print(f"  [WARN] {model_id!r} not found in local cache; downloading from HuggingFace...")
        return load_fn(model_id, local_files_only=False, **kwargs)
