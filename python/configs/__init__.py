# -*- coding: utf-8 -*-
"""
Central JSON-backed configuration for the Python pipeline.

This module loads static configuration from JSON files in the same directory
so that model definitions and prompt lists can be edited without touching code.
"""

import os
import json

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_json(name):
    """Load a JSON file from the configs directory."""
    path = os.path.join(_CONFIG_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


MODEL_CONFIGS = _load_json("model_configs.json")
DEFAULT_PROMPTS = _load_json("default_prompts.json")
