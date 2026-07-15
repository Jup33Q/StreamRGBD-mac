#!/bin/bash

# StreamRGBD Camera Inference Launcher
# Auto-activates venv and runs the RGBD pipeline

# Enable MPS fallback for unsupported ops (e.g. upsample_bicubic2d in DA3 on Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1

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
    --depth-backend auto \
    --ndi-output "StreamDiffusion-RGBD" \
    "$@"
