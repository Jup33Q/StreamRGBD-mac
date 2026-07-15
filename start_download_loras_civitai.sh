#!/bin/bash
# StreamDiffusion LoRA Downloader Launcher (Civitai)
# Downloads curated SD 1.5 compatible LoRAs from Civitai using API.
#
# Usage:
#   export CIVITAI_API_KEY=your_key_here
#   ./start_download_loras_civitai.sh
#
# Get your API key at: https://civitai.com/user/account

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv/bin" ]; then
    echo "ERROR: .venv not found. Run python/setup.sh first."
    exit 1
fi

source .venv/bin/activate

# Check API key from environment
if [ -z "$CIVITAI_API_KEY" ]; then
    echo "=========================================="
    echo "ERROR: CIVITAI_API_KEY is not set."
    echo "=========================================="
    echo ""
    echo "Please set your Civitai API Key before running:"
    echo "  export CIVITAI_API_KEY=your_api_key_here"
    echo ""
    echo "Get your free API key at: https://civitai.com/user/account"
    echo ""
    exit 1
fi

echo "=========================================="
echo "StreamDiffusion Civitai LoRA Downloader"
echo "=========================================="
echo ""

# Show help first
python python/download_loras_civitai.py --list

echo ""
echo "=========================================="
echo "Starting download of all LoRAs..."
echo "=========================================="
echo ""

# Download all LoRAs
python python/download_loras_civitai.py --all "$@"
