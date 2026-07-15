#!/bin/bash

# Camera → NDI 基础转发启动器
# 纯摄像头捕获 + NDI 推流，无 AI 处理

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv/bin" ]; then
    echo "ERROR: .venv 未找到。请先运行 python/setup.sh"
    exit 1
fi

source .venv/bin/activate

python python/camera_to_ndi.py \
    --camera 0 \
    --ndi-output "Camera-NDI" \
    --width 1280 \
    --height 720 \
    --fps 30 \
    "$@"
