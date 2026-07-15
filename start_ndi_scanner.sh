#!/bin/bash

# NDI Source Scanner GUI 启动器

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv/bin" ]; then
    echo "ERROR: .venv 未找到。请先运行 python/setup.sh"
    exit 1
fi

source .venv/bin/activate

python python/ndi_scanner_gui.py "$@"
