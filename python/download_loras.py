#!/usr/bin/env python3
"""
StreamDiffusion LoRA Batch Downloader
Downloads SD 1.5 compatible LoRAs from HuggingFace for use with streamdiffusion-mac.

Usage:
    python download_loras.py --all           # Download all recommended LoRAs
    python download_loras.py --list          # List available LoRAs
    python download_loras.py --name watercolor  # Download specific LoRA
    python download_loras.py --custom <repo_id> <filename>  # Download custom LoRA
"""
import os
import sys
import argparse
from pathlib import Path

# huggingface_hub is already in requirements.txt dependencies
try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("Installing huggingface_hub...")
    os.system(f"{sys.executable} -m pip install huggingface-hub")
    from huggingface_hub import hf_hub_download, list_repo_files


# Curated SD 1.5 compatible LoRAs for real-time style transfer
# All from HuggingFace (no Civitai API key needed)
LORA_REGISTRY = {
    "detail_tweaker": {
        "repo_id": "Linaqruf/add-detail-lora",
        "filename": "add_detail.safetensors",
        "description": "Universal detail enhancer. Positive=more detail, negative=simpler.",
        "weight_range": (-2.0, 2.0),
        "trigger_words": "",
    },
    "epi_noiseoffset": {
        "repo_id": "Linaqruf/epiCNoiseoffset",
        "filename": "epi_noiseoffset2.safetensors",
        "description": "Dramatic lighting with deep blacks and vibrant highlights.",
        "weight_range": (0.5, 1.0),
        "trigger_words": "",
    },
    "moxin_ink": {
        "repo_id": "Linaqruf/moxin-lora",
        "filename": "MoXinV1.safetensors",
        "description": "Chinese ink painting style (Shuimo/Bonian/Zhenbanqiao/Badashanren).",
        "weight_range": (0.6, 1.0),
        "trigger_words": "shuimobysim, wuchangshuo, bonian, zhenbanqiao, badashanren",
    },
    "anime_lineart": {
        "repo_id": "Linaqruf/anime-lineart-lora",
        "filename": "animeoutlineV4_16.safetensors",
        "description": "Clean lineart and manga-style illustrations.",
        "weight_range": (0.6, 1.0),
        "trigger_words": "lineart, monochrome",
    },
    "3d_render": {
        "repo_id": "Linaqruf/3d-render-style-lora",
        "filename": "3DMM_V12.safetensors",
        "description": "3D rendering style with realistic materials.",
        "weight_range": (0.6, 1.0),
        "trigger_words": "3d, realistic, 3DMM",
    },
    "tarot_card": {
        "repo_id": "Linaqruf/anime-tarot-lora",
        "filename": "animetarotV51.safetensors",
        "description": "Anime tarot card art with Mucha-like outlines.",
        "weight_range": (0.6, 1.0),
        "trigger_words": "tarot, anime tarot",
    },
    "pixel_art": {
        "repo_id": "Linaqruf/pixel-art-lora",
        "filename": "pixelartV3.safetensors",
        "description": "Retro pixel art / dot art style.",
        "weight_range": (0.7, 1.0),
        "trigger_words": "pixelart, pixel",
    },
    "pastel_color": {
        "repo_id": "Linaqruf/pastel-color-lora",
        "filename": "Pastel color.safetensors",
        "description": "Soft pastel palette, good for retro street scenes.",
        "weight_range": (0.5, 0.8),
        "trigger_words": "pastel color",
    },
    "sci_fi_env": {
        "repo_id": "Linaqruf/sci-fi-environment-lora",
        "filename": "Sci-fi_Enviroments.safetensors",
        "description": "Sci-fi environments and futuristic worlds.",
        "weight_range": (0.6, 1.0),
        "trigger_words": "sci-fi, futuristic",
    },
    "flat_illustration": {
        "repo_id": "Linaqruf/flat2-lora",
        "filename": "flat2.safetensors",
        "description": "Flat illustration style (positive) or more detail (negative).",
        "weight_range": (-1.0, 1.0),
        "trigger_words": "flat illustration",
    },
}


def get_loras_dir():
    """Get the LoRAs download directory."""
    script_dir = Path(__file__).parent.resolve()
    loras_dir = script_dir / "loras"
    loras_dir.mkdir(exist_ok=True)
    return loras_dir


def download_lora(name: str, repo_id: str, filename: str, loras_dir: Path, force: bool = False):
    """Download a single LoRA from HuggingFace."""
    output_path = loras_dir / f"{name}.safetensors"

    if output_path.exists() and not force:
        print(f"  [SKIP] {name}: already exists at {output_path}")
        return output_path

    print(f"  [DOWNLOAD] {name} from {repo_id}/{filename}")
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(loras_dir),
            local_dir_use_symlinks=False,
        )
        # Rename to consistent naming
        downloaded_path = Path(downloaded)
        if downloaded_path.name != f"{name}.safetensors":
            target = loras_dir / f"{name}.safetensors"
            if target.exists():
                target.unlink()
            downloaded_path.rename(target)
            downloaded = str(target)
        print(f"  [OK] {name} -> {downloaded}")
        return Path(downloaded)
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return None


def list_available():
    """Print all available LoRAs."""
    print("=" * 70)
    print("Available SD 1.5 LoRAs for StreamDiffusion")
    print("=" * 70)
    for name, info in LORA_REGISTRY.items():
        print(f"\n{name}")
        print(f"  Description: {info['description']}")
        print(f"  Weight:      {info['weight_range'][0]} ~ {info['weight_range'][1]}")
        print(f"  Triggers:    {info['trigger_words'] or '(none)'}")
        print(f"  Source:      {info['repo_id']}/{info['filename']}")
    print("\n" + "=" * 70)
    print(f"Total: {len(LORA_REGISTRY)} LoRAs")


def download_all(loras_dir: Path, force: bool = False):
    """Download all recommended LoRAs."""
    print(f"Downloading all LoRAs to: {loras_dir}")
    print(f"Total: {len(LORA_REGISTRY)} LoRAs\n")

    success = 0
    failed = 0
    for name, info in LORA_REGISTRY.items():
        result = download_lora(name, info["repo_id"], info["filename"], loras_dir, force)
        if result:
            success += 1
        else:
            failed += 1
        print()

    print("=" * 70)
    print(f"Download complete: {success} success, {failed} failed")
    print(f"LoRAs saved to: {loras_dir}")


def download_custom(repo_id: str, filename: str, output_name: str = None, loras_dir: Path = None, force: bool = False):
    """Download a custom LoRA from any HuggingFace repo."""
    name = output_name or filename.replace(".safetensors", "").replace(".", "_")
    return download_lora(name, repo_id, filename, loras_dir, force)


def scan_local_loras(loras_dir: Path):
    """List already downloaded LoRAs."""
    files = sorted(loras_dir.glob("*.safetensors"))
    print(f"Local LoRAs in {loras_dir}:")
    if not files:
        print("  (none)")
        return
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:40s}  {size_mb:6.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="StreamDiffusion LoRA Downloader")
    parser.add_argument("--list", action="store_true", help="List available LoRAs")
    parser.add_argument("--all", action="store_true", help="Download all recommended LoRAs")
    parser.add_argument("--name", type=str, help="Download specific LoRA by name")
    parser.add_argument("--custom", nargs=2, metavar=("REPO_ID", "FILENAME"),
                        help="Download custom LoRA: --custom <repo_id> <filename>")
    parser.add_argument("--output-name", type=str, help="Custom output name for --custom")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    parser.add_argument("--scan", action="store_true", help="List local LoRAs")
    args = parser.parse_args()

    loras_dir = get_loras_dir()

    if args.list:
        list_available()
        return

    if args.scan:
        scan_local_loras(loras_dir)
        return

    if args.all:
        download_all(loras_dir, force=args.force)
        return

    if args.name:
        if args.name not in LORA_REGISTRY:
            print(f"ERROR: Unknown LoRA '{args.name}'")
            print(f"Available: {', '.join(LORA_REGISTRY.keys())}")
            return
        info = LORA_REGISTRY[args.name]
        download_lora(args.name, info["repo_id"], info["filename"], loras_dir, force=args.force)
        return

    if args.custom:
        repo_id, filename = args.custom
        download_custom(repo_id, filename, args.output_name, loras_dir, force=args.force)
        return

    # Default: show help + list
    list_available()
    print("\nUse --all to download everything, or --name <name> for a specific one.")


if __name__ == "__main__":
    main()
