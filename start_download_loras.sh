#!/bin/bash
# StreamDiffusion LoRA Downloader Launcher
# Downloads all recommended SD 1.5 compatible LoRAs from HuggingFace
#
# Environment:
#   HUGGINGFACE_TOKEN - HuggingFace API token (optional, for private/gated models)

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv/bin" ]; then
    echo "ERROR: .venv not found. Run python/setup.sh first."
    exit 1
fi

source .venv/bin/activate

echo "=========================================="
echo "StreamDiffusion LoRA Downloader"
echo "=========================================="
echo ""

# Optional token info
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "[INFO] HUGGINGFACE_TOKEN is set."
else
    echo "[INFO] HUGGINGFACE_TOKEN not set. Set it for private/gated models:"
    echo "       export HUGGINGFACE_TOKEN=hf_xxxxxxxx"
fi
echo ""

# Show help first
python python/download_loras.py --list

echo ""
echo "=========================================="
echo "Starting download of all LoRAs..."
echo "=========================================="
echo ""

# Download all LoRAs
python python/download_loras.py --all "$@"
