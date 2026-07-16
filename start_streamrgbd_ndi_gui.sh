#!/bin/bash

# Stream NDI GUI 启动器（NDI 输入 → AI → NDI 输出）

# Enable MPS fallback for unsupported ops (e.g. upsample_bicubic2d in DA3 on Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv/bin" ]; then
    echo "ERROR: .venv 未找到。请先运行 python/setup.sh"
    exit 1
fi

source .venv/bin/activate

python python/stream_ndi_gui.py "$@"
