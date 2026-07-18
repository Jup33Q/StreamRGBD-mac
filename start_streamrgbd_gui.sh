#!/bin/bash

# Stream RGBD GUI 启动器

# Enable MPS fallback for unsupported ops (e.g. upsample_bicubic2d in DA3 on Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# huggingface.co 不可达时自动切换到 hf-mirror 镜像，避免联网重试长时间挂起
if ! curl -sI --max-time 3 https://huggingface.co >/dev/null 2>&1 && \
   curl -sI --max-time 3 https://hf-mirror.com >/dev/null 2>&1; then
    export HF_ENDPOINT=https://hf-mirror.com
    echo "[INFO] huggingface.co 不可达，已切换 HF_ENDPOINT=$HF_ENDPOINT"
fi

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv/bin" ]; then
    echo "ERROR: .venv 未找到。请先运行 python/setup.sh"
    exit 1
fi

source .venv/bin/activate

python python/stream_rgbd_gui_db.py "$@"
