#!/bin/bash

# StreamRGBD Camera Inference Launcher
# Auto-activates venv and runs the RGBD pipeline

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv/bin" ]; then
    echo "ERROR: .venv not found. Run python/setup.sh first."
    exit 1
fi

source .venv/bin/activate

python python/camera_rgbd.py \
    --prompt "oil painting style, masterpiece" \
    --render-size 512 \
    --depth-backend pytorch \
    --depth-model da2-small \
    --ndi-output "StreamDiffusion-RGBD" \
    "$@"
