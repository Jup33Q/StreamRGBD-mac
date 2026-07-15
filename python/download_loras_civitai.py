#!/usr/bin/env python3
"""
StreamDiffusion LoRA Batch Downloader (Civitai)
Downloads SD 1.5 compatible LoRAs from Civitai using API.

Usage:
    python download_loras_civitai.py --all           # Download all curated LoRAs
    python download_loras_civitai.py --list          # List available LoRAs
    python download_loras_civitai.py --name detail   # Download specific LoRA
    python download_loras_civitai.py --search "kimono" # Search Civitai for LoRAs
    python download_loras_civitai.py --download <model_id> <version_id>  # Custom download

Environment:
    CIVITAI_API_KEY  - Civitai API key (required)
"""
import os
import sys
import json
import struct
import argparse
import urllib.request
import urllib.error
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# API Configuration
# ─────────────────────────────────────────────────────────────────────
CIVITAI_API_BASE = "https://civitai.com/api/v1"

# 150MB 下载上限
MAX_DOWNLOAD_MB = 150.0


def _get_api_key() -> str:
    """从环境变量读取 Civitai API Key。"""
    key = os.environ.get("CIVITAI_API_KEY")
    if not key:
        print("ERROR: CIVITAI_API_KEY environment variable is not set.")
        print("Please set it before running this script:")
        print("  export CIVITAI_API_KEY=your_api_key_here")
        print("Get your key at: https://civitai.com/user/account")
        sys.exit(1)
    return key


CIVITAI_API_KEY = _get_api_key()


def api_request(path: str, params: dict = None) -> dict:
    """Make a Civitai API request and return JSON."""
    url = f"{CIVITAI_API_BASE}{path}"
    if params:
        query = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
        url = f"{url}?{query}"

    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {CIVITAI_API_KEY}")
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"[API ERROR] {e.code}: {e.reason}")
        try:
            err_body = e.read().decode("utf-8")
            print(f"  Response: {err_body[:500]}")
        except Exception:
            pass
        return {}
    except Exception as e:
        print(f"[ERROR] {e}")
        return {}


def get_model_info(model_id: int) -> dict:
    """Fetch model details from Civitai."""
    return api_request(f"/models/{model_id}")


def search_models(query: str = "", types: str = "LORA", base_models: str = "SD 1.5", limit: int = 20, nsfw: bool = False) -> list:
    """Search Civitai for models."""
    params = {
        "types": types,
        "limit": limit,
        "nsfw": "false" if not nsfw else "true",
    }
    if query:
        params["query"] = query
    if base_models:
        params["baseModels"] = base_models

    resp = api_request("/models", params)
    return resp.get("items", [])


# ═════════════════════════════════════════════════════════════════════
# Curated LoRA Registry (Civitai)
# 与 db_init.py 中 LORA_MODEL_SEEDS 的 civitai 来源条目保持一致
# ═════════════════════════════════════════════════════════════════════

LORA_REGISTRY = {
    # ── 风格 LoRA ──
    "civitai-detail-tweaker": {
        "model_id": 58390,
        "version_id": 62833,
        "display_name": "Detail Tweaker 细节增强",
        "filename": "add_detail.safetensors",
        "description": "通用细节增强器。正值增加细节，负值简化画面。适用于所有基础模型。",
        "weight_min": -2.0,
        "weight_max": 2.0,
        "weight_default": 0.5,
        "trigger_words": "",
        "category": "quality",
        "sub_type": "detail",
        "file_size_mb": 36.0,
    },
    "civitai-moxin-ink": {
        "model_id": 12597,
        "version_id": 14856,
        "display_name": "MoXin 墨心 中国水墨",
        "filename": "MoXinV1.safetensors",
        "description": "中国水墨画风格，融合书法与国画笔触。",
        "weight_min": 0.6,
        "weight_max": 1.0,
        "weight_default": 0.8,
        "trigger_words": "shuimobysim, wuchangshuo, bonian, zhenbanqiao, badashanren",
        "category": "style",
        "sub_type": "painting",
        "file_size_mb": 144.1,
    },
    "civitai-anime-lineart": {
        "model_id": 16014,
        "version_id": 28907,
        "display_name": "Anime Lineart 动漫线稿",
        "filename": "animeoutlineV4_16.safetensors",
        "description": "干净线稿和漫画风格插画。",
        "weight_min": 0.6,
        "weight_max": 1.0,
        "weight_default": 0.8,
        "trigger_words": "lineart, monochrome",
        "category": "style",
        "sub_type": "painting",
        "file_size_mb": 18.1,
    },
    "civitai-more-details": {
        "model_id": 82098,
        "version_id": 87153,
        "display_name": "More Details 更多细节",
        "filename": "more_details.safetensors",
        "description": "细节增强工具 LoRA，提升画面精细度和纹理表现。",
        "weight_min": 0.3,
        "weight_max": 1.0,
        "weight_default": 0.6,
        "trigger_words": "",
        "category": "quality",
        "sub_type": "detail",
        "file_size_mb": 36.0,
    },
    "civitai-studio-ghibli": {
        "model_id": 6526,
        "version_id": 7656,
        "display_name": "Studio Ghibli 吉卜力风格",
        "filename": "ghibli_style_offset.safetensors",
        "description": "吉卜力工作室动画风格，温暖色调和细腻背景。",
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "ghibli, studio ghibli",
        "category": "style",
        "sub_type": "animation",
        "file_size_mb": 144.0,
    },
    # ── 角色 / 服装 LoRA ──
    "civitai-hanfu": {
        "model_id": 15365,
        "version_id": 18104,
        "display_name": "Hanfu 汉服",
        "filename": "hanfu_v30.safetensors",
        "description": "中国传统汉服风格，魏晋、唐宋、明制多种形制。",
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "hanfu, chinese dress",
        "category": "subject",
        "sub_type": "clothing",
        "file_size_mb": 36.0,
    },
    "civitai-clothing-adjuster": {
        "model_id": 88132,
        "version_id": 93564,
        "display_name": "Clothing Adjuster 衣物调整",
        "filename": "ClothingAdjuster3.safetensors",
        "description": "控制人物服装增减，正向增加衣物，负向减少。",
        "weight_min": -1.0,
        "weight_max": 1.0,
        "weight_default": 0.5,
        "trigger_words": "",
        "category": "quality",
        "sub_type": "adjust",
        "file_size_mb": 4.6,
    },
    "civitai-moesode": {
        "model_id": 114494,
        "version_id": 123776,
        "display_name": "Moesode 萌袖",
        "filename": "moesode.safetensors",
        "description": "萌袖/袖过指服装风格，可爱袖子。",
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.7,
        "trigger_words": "moesode, sleeves past wrists",
        "category": "subject",
        "sub_type": "clothing",
        "file_size_mb": 36.1,
    },
    "civitai-instant-photo": {
        "model_id": 52652,
        "version_id": 57774,
        "display_name": "Instant Photo 拍立得",
        "filename": "INSPHOT2.safetensors",
        "description": "拍立得/Polaroid 照片风格，复古边框。",
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "instant photo, polaroid",
        "category": "style",
        "sub_type": "photo",
        "file_size_mb": 72.1,
    },
    # ── 建筑 LoRA ──
    "civitai-arch-sketch-markers": {
        "display_name": "Civitai Arch Sketch Markers 建筑马克笔草图",
        "filename": None,
        "description": "马克笔风格建筑草图，适合概念设计和手绘效果图（Civitai 来源）。",
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "marker sketch, architectural sketch",
        "category": "subject",
        "sub_type": "architecture",
        "file_size_mb": 9.1
    },
    "civitai-arch-sketch-style": {
        "display_name": "Civitai Arch Sketch Style 建筑线稿风格",
        "filename": None,
        "description": "建筑线稿和手绘风格，清晰的线条和透视表现（Civitai 来源）。",
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "archisketch, architectural sketch",
        "category": "subject",
        "sub_type": "architecture",
        "file_size_mb": 18.1
    },
    "civitai-arch-watercolor": {
        "display_name": "Civitai Arch Watercolor 建筑水彩风格",
        "filename": None,
        "description": "建筑水彩画风格，柔和的色彩和透明感（Civitai 来源）。",
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "watercolor, architecture watercolor",
        "category": "subject",
        "sub_type": "architecture",
        "file_size_mb": 36.1
    },
    "civitai-arch-concepts": {
        "display_name": "Civitai Arch Concepts 建筑场景单体",
        "filename": None,
        "description": "建筑场景单体概念设计，适合环境设计和场景搭建（Civitai 来源）。",
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "architectural concept, scene design",
        "category": "subject",
        "sub_type": "architecture",
        "file_size_mb": 36.1
    },
    "civitai-interior-design": {
        "display_name": "Civitai Interior Design 室内装潢设计",
        "filename": None,
        "description": "室内装潢设计风格，适合家居和室内空间渲染（Civitai 来源）。",
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "interior design, room design",
        "category": "subject",
        "sub_type": "architecture",
        "file_size_mb": 72.1
    },

}


def get_loras_dir():
    """Get the LoRAs download directory."""
    script_dir = Path(__file__).parent.resolve()
    loras_dir = script_dir / "loras"
    loras_dir.mkdir(exist_ok=True)
    return loras_dir


def _update_db_file_size(name: str, size_mb: float, category: str = None):
    """更新数据库中 LoRA 的 file_size_mb 字段（统一表 + 分类表）。"""
    try:
        current_dir = Path(__file__).parent.resolve()
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        from models import LoraModel, StyleLoraModel, SubjectLoraModel, QualityLoraModel

        # 更新统一表
        query = LoraModel.update(file_size_mb=round(size_mb, 2)).where(LoraModel.name == name)
        query.execute()

        # 更新分类表
        if category == "style":
            query = StyleLoraModel.update(file_size_mb=round(size_mb, 2)).where(StyleLoraModel.name == name)
            query.execute()
        elif category == "subject":
            query = SubjectLoraModel.update(file_size_mb=round(size_mb, 2)).where(SubjectLoraModel.name == name)
            query.execute()
        elif category == "quality":
            query = QualityLoraModel.update(file_size_mb=round(size_mb, 2)).where(QualityLoraModel.name == name)
            query.execute()
    except Exception:
        # DB 可能不存在或模型名不匹配，静默忽略
        pass


def download_lora(name: str, version_id: int, filename: str, loras_dir: Path,
                  force: bool = False, file_size_mb: float = None, category: str = None):
    """Download a single LoRA from Civitai."""
    output_path = loras_dir / f"{name}.safetensors"

    if output_path.exists() and not force:
        actual_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  [SKIP] {name}: already exists at {output_path} ({actual_size_mb:.1f}MB)")
        _update_db_file_size(name, actual_size_mb, category)
        return output_path

    # 100MB 跳过检查
    if file_size_mb is not None and file_size_mb > MAX_DOWNLOAD_MB:
        print(f"  [SKIP] {name}: estimated size {file_size_mb:.1f}MB exceeds {MAX_DOWNLOAD_MB}MB limit")
        return None

    download_url = f"https://civitai.com/api/download/models/{version_id}?type=Model&format=SafeTensor"
    print(f"  [DOWNLOAD] {name} from Civitai (v{version_id})")
    print(f"             URL: {download_url}")

    try:
        req = urllib.request.Request(download_url)
        req.add_header("Authorization", f"Bearer {CIVITAI_API_KEY}")
        req.add_header("Accept", "application/octet-stream,*/*")
        req.add_header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        req.add_header("Referer", "https://civitai.com/")

        with urllib.request.urlopen(req, timeout=300) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
            # 下载前再次检查大小
            if total_size > 0 and total_size / (1024 * 1024) > MAX_DOWNLOAD_MB:
                print(f"  [SKIP] {name}: actual size {total_size/1024/1024:.1f}MB exceeds {MAX_DOWNLOAD_MB}MB limit")
                return None

            downloaded = 0
            chunk_size = 65536

            with open(output_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        print(f"\r             Progress: {pct:.1f}% ({downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB)", end="", flush=True)
            print()  # newline after progress

        # Verify it's a valid safetensors file
        with open(output_path, "rb") as f:
            header_len_bytes = f.read(8)
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            if not (0 < header_len < 10_000_000):
                print(f"  [WARN] {name}: downloaded file may not be valid safetensors (header_len={header_len})")
            else:
                actual_size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"  [OK] {name} -> {output_path} ({actual_size_mb:.1f}MB)")
                _update_db_file_size(name, actual_size_mb, category)
        return output_path

    except Exception as e:
        print(f"\n  [FAIL] {name}: {e}")
        # Clean up partial file
        if output_path.exists():
            output_path.unlink()
        return None


def list_available():
    """Print all available LoRAs in registry, grouped by category."""
    print("=" * 70)
    print("Available SD 1.5 LoRAs from Civitai")
    print("=" * 70)

    # Group by category then sub_type
    groups = {"style": {}, "subject": {}, "quality": {}}
    for name, info in LORA_REGISTRY.items():
        cat = info.get("category", "subject")
        sub = info.get("sub_type", "other")
        if cat not in groups:
            groups[cat] = {}
        if sub not in groups[cat]:
            groups[cat][sub] = []
        groups[cat][sub].append((name, info))

    total_count = 0
    for cat in ("style", "subject", "quality"):
        subs = groups.get(cat, {})
        if not subs:
            continue
        print(f"\n--- {cat.upper()} LoRAs ---")
        cat_count = 0
        for sub in sorted(subs.keys()):
            items = subs[sub]
            print(f"\n  [{sub}]")
            for name, info in items:
                cat_count += 1
                total_count += 1
                size_str = f"{info.get('file_size_mb', '?')}MB"
                dl_status = "✓" if info.get("version_id") else "⚠ placeholder"
                print(f"    {dl_status} {name}")
                print(f"       Display:  {info['display_name']}")
                print(f"       Size:     {size_str}")
                print(f"       Weight:   {info['weight_min']} ~ {info['weight_max']} (default {info['weight_default']})")
                if info.get('trigger_words'):
                    print(f"       Triggers: {info['trigger_words']}")
        print(f"\n  ({cat_count} {cat} LoRAs)")

    print("\n" + "=" * 70)
    print(f"Total: {total_count} LoRAs")
    print(f"Max download size: {MAX_DOWNLOAD_MB}MB (larger models will be skipped)")
    """Print all available LoRAs in registry."""
    print("=" * 70)
    print("Available SD 1.5 LoRAs from Civitai")
    print("=" * 70)

    style_count = 0
    char_count = 0

    char_names = {
        "civitai-hanfu", "civitai-clothing-adjuster",
        "civitai-moesode", "civitai-instant-photo",
        "civitai-more-details",
    }

    print("\n--- Style LoRAs ---")
    for name, info in LORA_REGISTRY.items():
        is_char = name in char_names
        if not is_char:
            style_count += 1
            size_str = f"{info.get('file_size_mb', '?')}MB"
            print(f"\n{name}")
            print(f"  Display:     {info['display_name']}")
            print(f"  Description: {info['description']}")
            print(f"  Weight:      {info['weight_min']} ~ {info['weight_max']} (default {info['weight_default']})")
            print(f"  Triggers:    {info['trigger_words'] or '(none)'}")
            print(f"  Size:        {size_str}")
            print(f"  Civitai:     https://civitai.com/models/{info.get('model_id', 'N/A')}")

    print("\n--- Character / Clothing LoRAs ---")
    for name, info in LORA_REGISTRY.items():
        is_char = name in char_names
        if is_char:
            char_count += 1
            size_str = f"{info.get('file_size_mb', '?')}MB"
            print(f"\n{name}")
            print(f"  Display:     {info['display_name']}")
            print(f"  Description: {info['description']}")
            print(f"  Weight:      {info['weight_min']} ~ {info['weight_max']} (default {info['weight_default']})")
            print(f"  Triggers:    {info['trigger_words'] or '(none)'}")
            print(f"  Size:        {size_str}")
            print(f"  Civitai:     https://civitai.com/models/{info.get('model_id', 'N/A')}")

    print("\n" + "=" * 70)
    print(f"Total: {len(LORA_REGISTRY)} LoRAs ({style_count} style + {char_count} character/clothing)")
    print(f"Max download size: {MAX_DOWNLOAD_MB}MB (larger models will be skipped)")


def download_all(loras_dir: Path, force: bool = False):
    """Download all LoRAs from registry."""
    print(f"Downloading all Civitai LoRAs to: {loras_dir}")
    print(f"Total: {len(LORA_REGISTRY)} LoRAs")
    print(f"Skip threshold: {MAX_DOWNLOAD_MB}MB\n")

    success = 0
    skipped = 0
    failed = 0
    for name, info in LORA_REGISTRY.items():
        if info.get("version_id") is None or info.get("filename") is None:
            print(f"  [SKIP] {name}: missing version_id or filename (placeholder entry)")
            skipped += 1
            print()
            continue
        result = download_lora(name, info["version_id"], info["filename"], loras_dir,
                               force, info.get("file_size_mb"), info.get("category"))
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


def do_search(query: str, limit: int = 20):
    """Search Civitai for LoRAs."""
    print(f"Searching Civitai for '{query}' ...")
    items = search_models(query=query, limit=limit)
    print(f"Found {len(items)} results:\n")

    for item in items:
        model_id = item["id"]
        name = item["name"]
        creator = item["creator"]["username"]
        base_models = ", ".join(item.get("baseModels", []))
        stats = item.get("stats", {})
        downloads = stats.get("downloadCount", 0)
        thumbs_up = stats.get("thumbsUpCount", 0)

        versions = item.get("modelVersions", [])
        if versions:
            v = versions[0]
            v_id = v["id"]
            files = v.get("files", [])
            if files:
                fname = files[0]["name"]
                size_kb = files[0]["sizeKB"]
                size_mb = size_kb / 1024
                skip_flag = " [SKIP >150MB]" if size_mb > MAX_DOWNLOAD_MB else ""
                print(f"  ID: {model_id} | Version: {v_id}")
                print(f"  Name: {name}")
                print(f"  Creator: {creator} | Base: {base_models}")
                print(f"  File: {fname} ({size_mb:.1f}MB){skip_flag}")
                print(f"  Downloads: {downloads} | Likes: {thumbs_up}")
                print(f"  URL: https://civitai.com/models/{model_id}")
                print()


def main():
    parser = argparse.ArgumentParser(description="StreamDiffusion LoRA Downloader (Civitai)")
    parser.add_argument("--list", action="store_true", help="List available LoRAs in registry")
    parser.add_argument("--all", action="store_true", help="Download all LoRAs from registry")
    parser.add_argument("--name", type=str, help="Download specific LoRA by name")
    parser.add_argument("--search", type=str, metavar="QUERY", help="Search Civitai for LoRAs")
    parser.add_argument("--search-limit", type=int, default=20, help="Max search results (default 20)")
    parser.add_argument("--download", nargs=2, metavar=("MODEL_ID", "VERSION_ID"),
                        help="Download custom LoRA: --download <model_id> <version_id>")
    parser.add_argument("--output-name", type=str, help="Custom output name for --download")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    parser.add_argument("--scan", action="store_true", help="List local LoRAs")
    parser.add_argument("--api-key", type=str, help="Civitai API Key (or set CIVITAI_API_KEY env var)")
    args = parser.parse_args()

    global CIVITAI_API_KEY
    if args.api_key:
        CIVITAI_API_KEY = args.api_key

    if not CIVITAI_API_KEY:
        print("ERROR: Civitai API Key is required.")
        print("Get your key at: https://civitai.com/user/account")
        print("Usage: --api-key <key>  or  export CIVITAI_API_KEY=<key>")
        return

    loras_dir = get_loras_dir()

    if args.list:
        list_available()
        return

    if args.scan:
        scan_local_loras(loras_dir)
        return

    if args.search:
        do_search(args.search, limit=args.search_limit)
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
        if info.get("version_id") is None or info.get("filename") is None:
            print(f"ERROR: '{args.name}' is a placeholder entry with no Civitai download info.")
            print("Please find the actual model_id and version_id on Civitai, then use:")
            print(f"  python download_loras_civitai.py --download <model_id> <version_id> --output-name {args.name}")
            return
        download_lora(args.name, info["version_id"], info["filename"], loras_dir,
                      force=args.force, file_size_mb=info.get("file_size_mb"), category=info.get("category"))
        return

    if args.download:
        model_id, version_id = args.download
        # Try to get filename from API
        model_info = get_model_info(int(model_id))
        filename = f"model_{model_id}.safetensors"
        file_size_mb = None
        if model_info:
            for v in model_info.get("modelVersions", []):
                if str(v["id"]) == str(version_id):
                    files = v.get("files", [])
                    if files:
                        filename = files[0]["name"]
                        size_kb = files[0].get("sizeKB", 0)
                        file_size_mb = size_kb / 1024 if size_kb else None
                    break
        name = args.output_name or f"civitai_{model_id}"
        download_lora(name, int(version_id), filename, loras_dir,
                      force=args.force, file_size_mb=file_size_mb)
        return

    # Default: show help + list
    list_available()
    print("\nUse --all to download everything, or --name <name> for a specific one.")
    print("Use --search <query> to discover new LoRAs on Civitai.")


if __name__ == "__main__":
    main()
