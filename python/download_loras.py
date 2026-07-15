#!/usr/bin/env python3
"""
StreamDiffusion LoRA Batch Downloader
Downloads SD 1.5 compatible LoRAs from HuggingFace for use with streamdiffusion-mac.

Usage:
    python download_loras.py --all           # Download all recommended LoRAs
    python download_loras.py --list          # List available LoRAs
    python download_loras.py --name watercolor  # Download specific LoRA
    python download_loras.py --custom <repo_id> <filename>  # Download custom LoRA

Environment:
    HUGGINGFACE_TOKEN  - HuggingFace API token (optional, for private/gated models)
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


# ─────────────────────────────────────────────────────────────────────
# API Token
# ─────────────────────────────────────────────────────────────────────
def _get_hf_token() -> str | None:
    """从环境变量读取 HuggingFace Token。"""
    return os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")


HF_TOKEN = _get_hf_token()


# ═════════════════════════════════════════════════════════════════════
# Curated SD 1.5 compatible LoRAs for real-time style transfer
# All from HuggingFace (no Civitai API key needed)
# ═════════════════════════════════════════════════════════════════════
# 与 db_init.py 中 LORA_MODEL_SEEDS 的 huggingface 来源条目保持一致
# 已移除不存在的 Linaqruf LoRA

LORA_REGISTRY = {
    # ── 风格 LoRA ──
    "pixelart_redmond": {
        "repo_id": "artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
        "filename": "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors",
        "description": "SD 1.5 像素艺术 LoRA，crisp 复古游戏风格，有限调色板。",
        "weight_range": (0.6, 1.0),
        "weight_default": 0.8,
        "trigger_words": "pixel art, pixelart, retro game",
        "file_size_mb": 26.0,
    },
    "sketch_sd15": {
        "repo_id": "jordanhilado/sd-1-5-sketch-lora",
        "filename": "pytorch_lora_weights.safetensors",
        "description": "SD 1.5 铅笔素描风格 LoRA，松散手绘线条和可见笔触。",
        "weight_range": (0.5, 0.9),
        "weight_default": 0.7,
        "trigger_words": "sketch, pencil drawing, gestural",
        "file_size_mb": 3.1,
    },
    # ── 角色 / 服装 LoRA ──
    "waifu_diffusion": {
        "repo_id": "waifu-diffusion/wd-1-5-beta3",
        "filename": "wd-illusion-fp16.safetensors",
        "description": "Waifu Diffusion 动漫人物风格，适合日式二次元角色。",
        "weight_range": (0.5, 1.0),
        "weight_default": 0.8,
        "trigger_words": "anime girl, waifu, 1girl",
        "file_size_mb": 2460.5,
    },
    "victorian_dress": {
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Goth_Moon.safetensors",
        "description": "哥特与维多利亚风格服装，蕾丝、束腰、长裙。",
        "weight_range": (0.5, 1.0),
        "weight_default": 0.75,
        "trigger_words": "gothic dress, victorian, lace",
        "file_size_mb": 36.1,
    },
    "kimono_outfit": {
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Haruka.safetensors",
        "description": "日式和服与浴衣风格，樱花图案与腰带。",
        "weight_range": (0.5, 1.0),
        "weight_default": 0.75,
        "trigger_words": "kimono, yukata, obi, floral pattern",
        "file_size_mb": 36.1,
    },
    "cyberpunk_outfit": {
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Cyberpunk_style-05.safetensors",
        "description": "赛博朋克风格服装，霓虹装饰、皮夹克、发光线条。",
        "weight_range": (0.5, 1.0),
        "weight_default": 0.75,
        "trigger_words": "cyberpunk outfit, neon, leather jacket",
        "file_size_mb": 36.1,
    },
    "fantasy_armor": {
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Fantasy-10.safetensors",
        "description": "中世纪奇幻风格铠甲与盔甲，金属质感、龙鳞纹理。",
        "weight_range": (0.5, 1.0),
        "weight_default": 0.75,
        "trigger_words": "fantasy armor, plate armor, knight",
        "file_size_mb": 36.1,
    },
    "school_uniform": {
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Headphones.safetensors",
        "description": "日式学生制服风格（JK），水手服、领结、百褶裙。",
        "weight_range": (0.5, 1.0),
        "weight_default": 0.7,
        "trigger_words": "school uniform, seifuku, pleated skirt",
        "file_size_mb": 18.1,
    },
    "cheongsam_dress": {
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Fashion.safetensors",
        "description": "中式旗袍风格，高开叉、立领、刺绣花纹。",
        "weight_range": (0.5, 1.0),
        "weight_default": 0.75,
        "trigger_words": "cheongsam, qipao, chinese dress, embroidery",
        "file_size_mb": 36.1,
    },
    "maid_outfit": {
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "AiBunnies.safetensors",
        "description": "经典女仆装风格，蕾丝围裙、头饰、黑白配色。",
        "weight_range": (0.5, 1.0),
        "weight_default": 0.7,
        "trigger_words": "maid outfit, frilled apron, headdress",
        "file_size_mb": 36.1,
    },
    "military_uniform": {
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "DollarStore_Superman.safetensors",
        "description": "军装风格制服，肩章、勋章、迷彩元素。",
        "weight_range": (0.5, 1.0),
        "weight_default": 0.75,
        "trigger_words": "military uniform, epaulettes, medals",
        "file_size_mb": 18.1,
    },
    "wedding_dress": {
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Body_Positivity-05.safetensors",
        "description": "华丽婚纱风格，蕾丝拖尾、头纱、珍珠装饰。",
        "weight_range": (0.5, 1.0),
        "weight_default": 0.75,
        "trigger_words": "wedding dress, bridal gown, veil, lace",
        "file_size_mb": 36.1,
    },
    "ninja_outfit": {
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Hijikata_Toshizou.safetensors",
        "description": "日式忍者风格服装，蒙面、手里剑、暗黑配色。",
        "weight_range": (0.5, 1.0),
        "weight_default": 0.75,
        "trigger_words": "ninja outfit, shinobi, mask, dark clothes",
        "file_size_mb": 18.1,
    },
    "elven_cloak": {
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "FathomSDredo.safetensors",
        "description": "精灵风格斗篷与长袍，飘逸布料、自然色调、银饰。",
        "weight_range": (0.5, 1.0),
        "weight_default": 0.75,
        "trigger_words": "elven cloak, flowing robes, silver accessories",
        "file_size_mb": 36.1,
    },
    "pilot_suit": {
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "HotBoysv2.safetensors",
        "description": "飞行员/宇航员制服风格，皮夹克、护目镜、金属拉链。",
        "weight_range": (0.5, 1.0),
        "weight_default": 0.75,
        "trigger_words": "pilot suit, leather jacket, goggles, aviator",
        "file_size_mb": 18.1,
    },
    "pirate_costume": {
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "GambitVwhatever.safetensors",
        "description": "海盗风格服装，三角帽、眼罩、皮质腰带、长靴。",
        "weight_range": (0.5, 1.0),
        "weight_default": 0.75,
        "trigger_words": "pirate costume, tricorn hat, eyepatch, boots",
        "file_size_mb": 36.1,
    },
}

# 150MB 下载上限
MAX_DOWNLOAD_MB = 150.0


def get_loras_dir():
    """Get the LoRAs download directory."""
    script_dir = Path(__file__).parent.resolve()
    loras_dir = script_dir / "loras"
    loras_dir.mkdir(exist_ok=True)
    return loras_dir


def _update_db_file_size(name: str, size_mb: float):
    """更新数据库中 LoRA 的 file_size_mb 字段（如果数据库可用）。"""
    try:
        current_dir = Path(__file__).parent.resolve()
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        from models import LoraModel
        query = LoraModel.update(file_size_mb=round(size_mb, 2)).where(LoraModel.name == name.replace("_", "-"))
        query.execute()
    except Exception as e:
        # DB 可能不存在或模型名不匹配，静默忽略
        pass


def download_lora(name: str, repo_id: str, filename: str, loras_dir: Path,
                  force: bool = False, file_size_mb: float = None):
    """Download a single LoRA from HuggingFace."""
    output_path = loras_dir / f"{name}.safetensors"

    if output_path.exists() and not force:
        actual_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  [SKIP] {name}: already exists at {output_path} ({actual_size_mb:.1f}MB)")
        _update_db_file_size(name, actual_size_mb)
        return output_path

    # 100MB 跳过检查
    if file_size_mb is not None and file_size_mb > MAX_DOWNLOAD_MB:
        print(f"  [SKIP] {name}: estimated size {file_size_mb:.1f}MB exceeds {MAX_DOWNLOAD_MB}MB limit")
        return None

    print(f"  [DOWNLOAD] {name} from {repo_id}/{filename}")
    try:
        kwargs = {
            "repo_id": repo_id,
            "filename": filename,
            "local_dir": str(loras_dir),
            "local_dir_use_symlinks": False,
        }
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
            print(f"             Using HUGGINGFACE_TOKEN from environment")

        downloaded = hf_hub_download(**kwargs)
        # Rename to consistent naming
        downloaded_path = Path(downloaded)
        if downloaded_path.name != f"{name}.safetensors":
            target = loras_dir / f"{name}.safetensors"
            if target.exists():
                target.unlink()
            downloaded_path.rename(target)
            downloaded = str(target)

        actual_size_mb = Path(downloaded).stat().st_size / (1024 * 1024)
        print(f"  [OK] {name} -> {downloaded} ({actual_size_mb:.1f}MB)")
        _update_db_file_size(name, actual_size_mb)
        return Path(downloaded)
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return None


def list_available():
    """Print all available LoRAs."""
    print("=" * 70)
    print("Available SD 1.5 LoRAs for StreamDiffusion (HuggingFace)")
    print("=" * 70)

    style_count = 0
    char_count = 0

    print("\n--- Style LoRAs ---")
    for name, info in LORA_REGISTRY.items():
        is_char = info["repo_id"] == "EarthnDusk/Loras_2023" or name == "waifu_diffusion"
        if not is_char:
            style_count += 1
            size_str = f"{info.get('file_size_mb', '?')}MB"
            print(f"\n{name}")
            print(f"  Description: {info['description']}")
            print(f"  Weight:      {info['weight_range'][0]} ~ {info['weight_range'][1]} (default {info['weight_default']})")
            print(f"  Triggers:    {info['trigger_words'] or '(none)'}")
            print(f"  Source:      {info['repo_id']}/{info['filename']}")
            print(f"  Size:        {size_str}")

    print("\n--- Character / Clothing LoRAs ---")
    for name, info in LORA_REGISTRY.items():
        is_char = info["repo_id"] == "EarthnDusk/Loras_2023" or name == "waifu_diffusion"
        if is_char:
            char_count += 1
            size_str = f"{info.get('file_size_mb', '?')}MB"
            print(f"\n{name}")
            print(f"  Description: {info['description']}")
            print(f"  Weight:      {info['weight_range'][0]} ~ {info['weight_range'][1]} (default {info['weight_default']})")
            print(f"  Triggers:    {info['trigger_words'] or '(none)'}")
            print(f"  Source:      {info['repo_id']}/{info['filename']}")
            print(f"  Size:        {size_str}")

    print("\n" + "=" * 70)
    print(f"Total: {len(LORA_REGISTRY)} LoRAs ({style_count} style + {char_count} character/clothing)")
    print(f"Max download size: {MAX_DOWNLOAD_MB}MB (larger models will be skipped)")
    if HF_TOKEN:
        print("HuggingFace Token: loaded from environment")
    else:
        print("HuggingFace Token: not set (set HUGGINGFACE_TOKEN env var for private/gated models)")


def download_all(loras_dir: Path, force: bool = False):
    """Download all recommended LoRAs."""
    print(f"Downloading all LoRAs to: {loras_dir}")
    print(f"Total: {len(LORA_REGISTRY)} LoRAs")
    print(f"Skip threshold: {MAX_DOWNLOAD_MB}MB\n")

    success = 0
    skipped = 0
    failed = 0
    for name, info in LORA_REGISTRY.items():
        result = download_lora(name, info["repo_id"], info["filename"], loras_dir,
                               force, info.get("file_size_mb"))
        if result:
            success += 1
        elif info.get("file_size_mb", 0) > MAX_DOWNLOAD_MB:
            skipped += 1
        else:
            failed += 1
        print()

    print("=" * 70)
    print(f"Download complete: {success} success, {skipped} skipped (>{MAX_DOWNLOAD_MB}MB), {failed} failed")
    print(f"LoRAs saved to: {loras_dir}")


def download_custom(repo_id: str, filename: str, output_name: str = None,
                    loras_dir: Path = None, force: bool = False):
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
    total_size = 0
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {f.name:45s}  {size_mb:6.1f} MB")
    print(f"\n  Total: {len(files)} files, {total_size:.1f} MB")


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
    parser.add_argument("--token", type=str, help="HuggingFace API token (or set HUGGINGFACE_TOKEN env var)")
    args = parser.parse_args()

    global HF_TOKEN
    if args.token:
        HF_TOKEN = args.token

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
        download_lora(args.name, info["repo_id"], info["filename"], loras_dir,
                      force=args.force, file_size_mb=info.get("file_size_mb"))
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
