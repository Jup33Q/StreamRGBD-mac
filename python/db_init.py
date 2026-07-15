# -*- coding: utf-8 -*-
"""
StreamDiffusion macOS 项目数据库初始化脚本
负责：创建 data 目录、建表、插入种子数据、启用 WAL 模式
"""

import os
import json
import sys

# ─────────────────────────────────────────────────────────────
# 确保能导入同目录下的 models.py
# ─────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from models import (
    database,               # SqliteDatabase 实例
    StreamDiffusionModel,   # SD 模型表
    DepthModel,             # 深度模型表
    LoraModel,              # LoRA 模型表
    PromptCategory,         # 提示词类别表
    StylePrompt,            # 风格提示词表
    SubjectPrompt,          # 主题提示词表
    QualityPrompt,          # 质量提示词表
    AppSettings,            # 应用设置表
    BaseModel,              # 抽象基类（用于批量建表）
)

# ─────────────────────────────────────────────────────────────
# 数据库文件路径（与 models.py 中保持一致）
# ─────────────────────────────────────────────────────────────
DB_PATH = os.path.join(
    os.path.dirname(__file__),  # python/
    "..",                       # 项目根目录
    "data",                     # data/
    "streamdiffusion.db"         # 数据库文件
)

DATA_DIR = os.path.dirname(DB_PATH)


# ═════════════════════════════════════════════════════════════
# 种子数据定义
# ═════════════════════════════════════════════════════════════

# ── StreamDiffusionModel 种子数据 ──
STREAMDIFFUSION_MODEL_SEEDS = [
    {
        "name": "sdxs",
        "display_name": "SDXS 512 实时生成",
        "file_path": "models/sdxs-512-dpmsolver-8step.onnx",
        "model_kind": "t2i",
        "description": "SDXS 512 分辨率 8-step DPM-Solver 实时生成模型，专为 macOS 优化。",
        "parameters": json.dumps({
            "width": 512,
            "height": 512,
            "steps": 8,
            "cfg_scale": 1.0,
            "scheduler": "dpm-solver",
        }),
        "is_default": True,
        "is_active": True,
    },
    {
        "name": "sdxs-768",
        "display_name": "SDXS 768 高分辨率",
        "file_path": "models/sdxs-768-dpmsolver-8step.onnx",
        "model_kind": "t2i",
        "description": "SDXS 768 分辨率高画质实时生成模型，适合细节丰富的场景。",
        "parameters": json.dumps({
            "width": 768,
            "height": 768,
            "steps": 8,
            "cfg_scale": 1.0,
            "scheduler": "dpm-solver",
        }),
        "is_default": False,
        "is_active": True,
    },
    {
        "name": "sd-turbo",
        "display_name": "SD-Turbo 512 快速转换",
        "file_path": "models/sd-turbo-512.onnx",
        "model_kind": "img2img",
        "description": "Stable Diffusion Turbo 模型，1-4 步即可完成图像到图像转换，极快速度。",
        "parameters": json.dumps({
            "width": 512,
            "height": 512,
            "steps": 1,
            "cfg_scale": 1.0,
            "strength": 0.5,
        }),
        "is_default": False,
        "is_active": True,
    },
    {
        "name": "sd-turbo-768",
        "display_name": "SD-Turbo 768 高分辨率",
        "file_path": "models/sd-turbo-768.onnx",
        "model_kind": "img2img",
        "description": "SD-Turbo 768 分辨率图像转换模型，保留更多细节。",
        "parameters": json.dumps({
            "width": 768,
            "height": 768,
            "steps": 2,
            "cfg_scale": 1.0,
            "strength": 0.5,
        }),
        "is_default": False,
        "is_active": True,
    },
]

# ── DepthModel 种子数据 ──
DEPTH_MODEL_SEEDS = [
    {
        "name": "midas-depth",
        "display_name": "MiDaS 深度估计",
        "file_path": "models/midas-depth.onnx",
        "description": "MiDaS 单目深度估计模型，从单张图像生成高质量深度图。",
        "parameters": json.dumps({
            "model_type": "DPT_Large",
            "input_size": 384,
        }),
        "is_default": True,
        "is_active": True,
    },
    {
        "name": "depth-anything",
        "display_name": "Depth Anything 深度估计",
        "file_path": "models/depth-anything.onnx",
        "description": "Depth Anything 深度估计模型，对任意图像生成高质量深度图。",
        "parameters": json.dumps({
            "encoder": "vitl",
            "input_size": 518,
        }),
        "is_default": False,
        "is_active": True,
    },
]

# ── LoraModel 种子数据 ──
# 包含风格、角色、服装三类 LoRA
LORA_MODEL_SEEDS = [
    # ── 风格 LoRA ──
    {
        "name": "pixelart-redmond",
        "display_name": "PixelArt Redmond 像素艺术",
        "file_path": "loras/pixelart_redmond.safetensors",
        "repo_id": "artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
        "filename": "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors",
        "description": "SD 1.5 像素艺术 LoRA，crisp 复古游戏风格，有限调色板。",
        "parameters": json.dumps({'lora_scale': 0.8}),
        "weight_min": 0.6,
        "weight_max": 1.0,
        "weight_default": 0.8,
        "trigger_words": "pixel art, pixelart, retro game",
        "source_type": "huggingface",
        "file_size_mb": 26.0,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "sketch-sd15",
        "display_name": "Sketch SD 1.5 素描",
        "file_path": "loras/sketch_sd15.safetensors",
        "repo_id": "jordanhilado/sd-1-5-sketch-lora",
        "filename": "pytorch_lora_weights.safetensors",
        "description": "SD 1.5 铅笔素描风格 LoRA，松散手绘线条和可见笔触。",
        "parameters": json.dumps({'lora_scale': 0.7}),
        "weight_min": 0.5,
        "weight_max": 0.9,
        "weight_default": 0.7,
        "trigger_words": "sketch, pencil drawing, gestural",
        "source_type": "huggingface",
        "file_size_mb": 3.1,
        "is_default": False,
        "is_active": True
    },
    # ── 角色 / 服装 LoRA ──
    {
        "name": "waifu-diffusion",
        "display_name": "Waifu Diffusion 动漫人物",
        "file_path": "loras/waifu_diffusion.safetensors",
        "repo_id": "waifu-diffusion/wd-1-5-beta3",
        "filename": "wd-illusion-fp16.safetensors",
        "description": "Waifu Diffusion 动漫人物风格，适合日式二次元角色。",
        "parameters": json.dumps({'lora_scale': 0.8}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.8,
        "trigger_words": "anime girl, waifu, 1girl",
        "source_type": "huggingface",
        "file_size_mb": 2460.5,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "victorian-dress",
        "display_name": "Victorian Dress 维多利亚礼服",
        "file_path": "loras/victorian_dress.safetensors",
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Goth_Moon.safetensors",
        "description": "哥特与维多利亚风格服装，蕾丝、束腰、长裙。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "gothic dress, victorian, lace",
        "source_type": "huggingface",
        "file_size_mb": 36.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "kimono-outfit",
        "display_name": "Kimono Outfit 和服",
        "file_path": "loras/kimono_outfit.safetensors",
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Haruka.safetensors",
        "description": "日式和服与浴衣风格，樱花图案与腰带。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "kimono, yukata, obi, floral pattern",
        "source_type": "huggingface",
        "file_size_mb": 36.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "cyberpunk-outfit",
        "display_name": "Cyberpunk Outfit 赛博朋克服装",
        "file_path": "loras/cyberpunk_outfit.safetensors",
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Cyberpunk_style-05.safetensors",
        "description": "赛博朋克风格服装，霓虹装饰、皮夹克、发光线条。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "cyberpunk outfit, neon, leather jacket",
        "source_type": "huggingface",
        "file_size_mb": 36.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "fantasy-armor",
        "display_name": "Fantasy Armor 奇幻铠甲",
        "file_path": "loras/fantasy_armor.safetensors",
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Fantasy-10.safetensors",
        "description": "中世纪奇幻风格铠甲与盔甲，金属质感、龙鳞纹理。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "fantasy armor, plate armor, knight",
        "source_type": "huggingface",
        "file_size_mb": 36.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "school-uniform",
        "display_name": "School Uniform JK制服",
        "file_path": "loras/school_uniform.safetensors",
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Headphones.safetensors",
        "description": "日式学生制服风格（JK），水手服、领结、百褶裙。",
        "parameters": json.dumps({'lora_scale': 0.7}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.7,
        "trigger_words": "school uniform, seifuku, pleated skirt",
        "source_type": "huggingface",
        "file_size_mb": 18.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "cheongsam-dress",
        "display_name": "Cheongsam 旗袍",
        "file_path": "loras/cheongsam_dress.safetensors",
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Fashion.safetensors",
        "description": "中式旗袍风格，高开叉、立领、刺绣花纹。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "cheongsam, qipao, chinese dress, embroidery",
        "source_type": "huggingface",
        "file_size_mb": 36.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "maid-outfit",
        "display_name": "Maid Outfit 女仆装",
        "file_path": "loras/maid_outfit.safetensors",
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "AiBunnies.safetensors",
        "description": "经典女仆装风格，蕾丝围裙、头饰、黑白配色。",
        "parameters": json.dumps({'lora_scale': 0.7}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.7,
        "trigger_words": "maid outfit, frilled apron, headdress",
        "source_type": "huggingface",
        "file_size_mb": 36.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "military-uniform",
        "display_name": "Military Uniform 军装",
        "file_path": "loras/military_uniform.safetensors",
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "DollarStore_Superman.safetensors",
        "description": "军装风格制服，肩章、勋章、迷彩元素。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "military uniform, epaulettes, medals",
        "source_type": "huggingface",
        "file_size_mb": 18.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "wedding-dress",
        "display_name": "Wedding Dress 婚纱",
        "file_path": "loras/wedding_dress.safetensors",
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Body_Positivity-05.safetensors",
        "description": "华丽婚纱风格，蕾丝拖尾、头纱、珍珠装饰。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "wedding dress, bridal gown, veil, lace",
        "source_type": "huggingface",
        "file_size_mb": 36.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "ninja-outfit",
        "display_name": "Ninja Outfit 忍者装",
        "file_path": "loras/ninja_outfit.safetensors",
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "Hijikata_Toshizou.safetensors",
        "description": "日式忍者风格服装，蒙面、手里剑、暗黑配色。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "ninja outfit, shinobi, mask, dark clothes",
        "source_type": "huggingface",
        "file_size_mb": 18.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "elven-cloak",
        "display_name": "Elven Cloak 精灵斗篷",
        "file_path": "loras/elven_cloak.safetensors",
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "FathomSDredo.safetensors",
        "description": "精灵风格斗篷与长袍，飘逸布料、自然色调、银饰。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "elven cloak, flowing robes, silver accessories",
        "source_type": "huggingface",
        "file_size_mb": 36.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "pilot-suit",
        "display_name": "Pilot Suit 飞行员制服",
        "file_path": "loras/pilot_suit.safetensors",
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "HotBoysv2.safetensors",
        "description": "飞行员/宇航员制服风格，皮夹克、护目镜、金属拉链。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "pilot suit, leather jacket, goggles, aviator",
        "source_type": "huggingface",
        "file_size_mb": 18.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "pirate-costume",
        "display_name": "Pirate Costume 海盗装",
        "file_path": "loras/pirate_costume.safetensors",
        "repo_id": "EarthnDusk/Loras_2023",
        "filename": "GambitVwhatever.safetensors",
        "description": "海盗风格服装，三角帽、眼罩、皮质腰带、长靴。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "pirate costume, tricorn hat, eyepatch, boots",
        "source_type": "huggingface",
        "file_size_mb": 36.1,
        "is_default": False,
        "is_active": True
    },
    # ── Civitai 来源 LoRA ──
    {
        "name": "civitai-detail-tweaker",
        "display_name": "Civitai Detail Tweaker 细节增强",
        "file_path": "loras/detail-tweaker.safetensors",
        "repo_id": None,
        "filename": None,
        "civitai_version_id": 62833,
        "description": "通用细节增强器（Civitai 来源）。正值增加细节，负值简化画面。",
        "parameters": json.dumps({'lora_scale': 0.5}),
        "weight_min": -2.0,
        "weight_max": 2.0,
        "weight_default": 0.5,
        "trigger_words": "",
        "source_type": "civitai",
        "file_size_mb": 36.0,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "civitai-moxin-ink",
        "display_name": "Civitai MoXin 墨心 中国水墨",
        "file_path": "loras/moxin-ink.safetensors",
        "repo_id": None,
        "filename": None,
        "civitai_version_id": 14856,
        "description": "中国水墨画风格（Civitai 来源）。",
        "parameters": json.dumps({'lora_scale': 0.8}),
        "weight_min": 0.6,
        "weight_max": 1.0,
        "weight_default": 0.8,
        "trigger_words": "shuimobysim, wuchangshuo, bonian, zhenbanqiao, badashanren",
        "source_type": "civitai",
        "file_size_mb": 144.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "civitai-anime-lineart",
        "display_name": "Civitai Anime Lineart 动漫线稿",
        "file_path": "loras/anime-lineart.safetensors",
        "repo_id": None,
        "filename": None,
        "civitai_version_id": 28907,
        "description": "干净线稿和漫画风格插画（Civitai 来源）。",
        "parameters": json.dumps({'lora_scale': 0.8}),
        "weight_min": 0.6,
        "weight_max": 1.0,
        "weight_default": 0.8,
        "trigger_words": "lineart, monochrome",
        "source_type": "civitai",
        "file_size_mb": 18.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "civitai-more-details",
        "display_name": "Civitai More Details 更多细节",
        "file_path": "loras/more-details.safetensors",
        "repo_id": None,
        "filename": None,
        "civitai_version_id": 87153,
        "description": "细节增强工具 LoRA，提升画面精细度和纹理表现（Civitai 来源）。",
        "parameters": json.dumps({'lora_scale': 0.6}),
        "weight_min": 0.3,
        "weight_max": 1.0,
        "weight_default": 0.6,
        "trigger_words": "",
        "source_type": "civitai",
        "file_size_mb": 36.0,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "civitai-studio-ghibli",
        "display_name": "Civitai Studio Ghibli 吉卜力风格",
        "file_path": "loras/studio-ghibli.safetensors",
        "repo_id": None,
        "filename": None,
        "civitai_version_id": 7656,
        "description": "吉卜力工作室动画风格，温暖色调和细腻背景（Civitai 来源）。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "ghibli, studio ghibli",
        "source_type": "civitai",
        "file_size_mb": 144.0,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "civitai-hanfu",
        "display_name": "Civitai Hanfu 汉服",
        "file_path": "loras/hanfu.safetensors",
        "repo_id": None,
        "filename": None,
        "civitai_version_id": 18104,
        "description": "中国传统汉服风格，魏晋、唐宋、明制多种形制（Civitai 来源）。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "hanfu, chinese dress",
        "source_type": "civitai",
        "file_size_mb": 36.0,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "civitai-clothing-adjuster",
        "display_name": "Civitai Clothing Adjuster 衣物调整",
        "file_path": "loras/clothing-adjuster.safetensors",
        "repo_id": None,
        "filename": None,
        "civitai_version_id": 93564,
        "description": "控制人物服装增减，正向增加衣物，负向减少（Civitai 来源）。",
        "parameters": json.dumps({'lora_scale': 0.5}),
        "weight_min": -1.0,
        "weight_max": 1.0,
        "weight_default": 0.5,
        "trigger_words": "",
        "source_type": "civitai",
        "file_size_mb": 4.6,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "civitai-moesode",
        "display_name": "Civitai Moesode 萌袖",
        "file_path": "loras/moesode.safetensors",
        "repo_id": None,
        "filename": None,
        "civitai_version_id": 123776,
        "description": "萌袖/袖过指服装风格，可爱袖子（Civitai 来源）。",
        "parameters": json.dumps({'lora_scale': 0.7}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.7,
        "trigger_words": "moesode, sleeves past wrists",
        "source_type": "civitai",
        "file_size_mb": 36.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "civitai-instant-photo",
        "display_name": "Civitai Instant Photo 拍立得",
        "file_path": "loras/instant-photo.safetensors",
        "repo_id": None,
        "filename": None,
        "civitai_version_id": 57774,
        "description": "拍立得/Polaroid 照片风格，复古边框（Civitai 来源）。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "instant photo, polaroid",
        "source_type": "civitai",
        "file_size_mb": 72.1,
        "is_default": False,
        "is_active": True
    },
    # ── 建筑 LoRA ──
    {
        "name": "civitai-arch-sketch-markers",
        "display_name": "Civitai Arch Sketch Markers 建筑马克笔草图",
        "file_path": "loras/arch-sketch-markers.safetensors",
        "repo_id": None,
        "filename": None,
        "civitai_version_id": 40665,
        "description": "马克笔风格建筑草图，适合概念设计和手绘效果图（Civitai 来源）。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "marker sketch, architectural sketch",
        "source_type": "civitai",
        "file_size_mb": 9.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "civitai-arch-sketch-style",
        "display_name": "Civitai Arch Sketch Style 建筑线稿风格",
        "file_path": "loras/arch-sketch-style.safetensors",
        "repo_id": None,
        "filename": None,
        "civitai_version_id": 39140,
        "description": "建筑线稿和手绘风格，清晰的线条和透视表现（Civitai 来源）。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "archisketch, architectural sketch",
        "source_type": "civitai",
        "file_size_mb": 18.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "civitai-arch-watercolor",
        "display_name": "Civitai Arch Watercolor 建筑水彩风格",
        "file_path": "loras/arch-watercolor.safetensors",
        "repo_id": None,
        "filename": None,
        "civitai_version_id": 43656,
        "description": "建筑水彩画风格，柔和的色彩和透明感（Civitai 来源）。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "watercolor, architecture watercolor",
        "source_type": "civitai",
        "file_size_mb": 36.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "civitai-arch-concepts",
        "display_name": "Civitai Arch Concepts 建筑场景单体",
        "file_path": "loras/arch-concepts.safetensors",
        "repo_id": None,
        "filename": None,
        "civitai_version_id": 147752,
        "description": "建筑场景单体概念设计，适合环境设计和场景搭建（Civitai 来源）。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "architectural concept, scene design",
        "source_type": "civitai",
        "file_size_mb": 36.1,
        "is_default": False,
        "is_active": True
    },
    {
        "name": "civitai-interior-design",
        "display_name": "Civitai Interior Design 室内装潢设计",
        "file_path": "loras/interior-design.safetensors",
        "repo_id": None,
        "filename": None,
        "civitai_version_id": 43473,
        "description": "室内装潢设计风格，适合家居和室内空间渲染（Civitai 来源）。",
        "parameters": json.dumps({'lora_scale': 0.75}),
        "weight_min": 0.5,
        "weight_max": 1.0,
        "weight_default": 0.75,
        "trigger_words": "interior design, room design",
        "source_type": "civitai",
        "file_size_mb": 72.1,
        "is_default": False,
        "is_active": True
    },
]
# ── PromptCategory 种子数据 ──
PROMPT_CATEGORY_SEEDS = [
    {
        "name": "style",
        "display_name": "风格",
        "description": "图像风格提示词，如油画、水彩、赛博朋克等",
        "table_name": "style_prompt",
        "sort_order": 1,
        "is_active": True,
    },
    {
        "name": "subject",
        "display_name": "主题",
        "description": "图像主题提示词，如人物、风景、物体等",
        "table_name": "subject_prompt",
        "sort_order": 2,
        "is_active": True,
    },
    {
        "name": "quality",
        "display_name": "质量",
        "description": "图像质量修饰词，如 masterpiece、ultra detailed 等",
        "table_name": "quality_prompt",
        "sort_order": 3,
        "is_active": True,
    },
]

# ── StylePrompt 种子数据（18条）──
STYLE_PROMPT_SEEDS = [
    {
        "prompt_text": "oil painting, classical portrait, rich textures, warm golden lighting, Rembrandt style, masterpiece, highly detailed",
        "tags": "oil painting, classical, portrait, Rembrandt, masterpiece",
        "is_active": True,
    },
    {
        "prompt_text": "watercolor painting, soft pastel colors, delicate brushstrokes, dreamy atmosphere, ethereal light, artistic illustration",
        "tags": "watercolor, pastel, soft, dreamy, artistic",
        "is_active": True,
    },
    {
        "prompt_text": "cyberpunk cityscape, neon lights, rain-soaked streets, holographic advertisements, dystopian future, blade runner aesthetic, 8k ultra detailed",
        "tags": "cyberpunk, neon, rain, dystopian, blade runner, 8k",
        "is_active": True,
    },
    {
        "prompt_text": "anime style, vibrant colors, cel shading, Studio Ghibli inspired, whimsical scenery, magical atmosphere, detailed background",
        "tags": "anime, cel shading, Ghibli, vibrant, magical",
        "is_active": True,
    },
    {
        "prompt_text": "photorealistic portrait, natural lighting, skin texture detail, shallow depth of field, professional photography, 85mm lens, bokeh background",
        "tags": "photorealistic, portrait, natural light, professional, 85mm, bokeh",
        "is_active": True,
    },
    {
        "prompt_text": "impressionist painting, dappled sunlight, visible brushstrokes, Monet style, garden scene, soft focus, vibrant color palette",
        "tags": "impressionist, Monet, garden, sunlight, brushstrokes",
        "is_active": True,
    },
    {
        "prompt_text": "pixel art, retro game style, 8-bit aesthetic, vibrant limited palette, isometric view, detailed environment, nostalgic feeling",
        "tags": "pixel art, retro, 8-bit, isometric, nostalgic",
        "is_active": True,
    },
    {
        "prompt_text": "3D render, octane render, subsurface scattering, realistic materials, studio lighting, product photography style, clean background",
        "tags": "3D render, octane, realistic, studio lighting, product",
        "is_active": True,
    },
    # ── 新增风格提示词 ──
    {
        "prompt_text": "gothic dark fantasy, ornate baroque architecture, dramatic chiaroscuro shadows, crimson velvet, candlelit atmosphere, haunting beauty",
        "tags": "gothic, dark fantasy, baroque, chiaroscuro, candlelit",
        "is_active": True,
    },
    {
        "prompt_text": "ukiyo-e woodblock print, Hokusai style, bold outlines, flat color planes, Mount Fuji in background, traditional Japanese aesthetic, waves and clouds",
        "tags": "ukiyo-e, Hokusai, woodblock, Japanese, traditional",
        "is_active": True,
    },
    {
        "prompt_text": "steampunk brass gears, Victorian machinery, warm copper tones, leather and rivets, clockwork mechanisms, sepia tones, industrial elegance",
        "tags": "steampunk, brass, Victorian, clockwork, industrial",
        "is_active": True,
    },
    {
        "prompt_text": "neon cyberpunk, electric blue and magenta, rain-slicked streets reflecting neon signs, holographic displays, dystopian nightlife, vaporwave aesthetic",
        "tags": "neon, cyberpunk, electric, vaporwave, nightlife",
        "is_active": True,
    },
    {
        "prompt_text": "vintage film noir, high contrast black and white, cigarette smoke wisps, Venetian blinds shadows, femme fatale atmosphere, 1940s Hollywood",
        "tags": "film noir, black and white, 1940s, Hollywood, shadows",
        "is_active": True,
    },
    {
        "prompt_text": "pop art bold colors, Lichtenstein style, Ben-Day dots, comic panel aesthetic, saturated primary colors, graphic illustration, retro advertising",
        "tags": "pop art, Lichtenstein, comic, bold, graphic",
        "is_active": True,
    },
    {
        "prompt_text": "art nouveau flowing lines, Alphonse Mucha style, botanical motifs, ornate decorative borders, gold leaf accents, ethereal feminine beauty",
        "tags": "art nouveau, Mucha, botanical, decorative, gold leaf",
        "is_active": True,
    },
    {
        "prompt_text": "minimalist clean lines, negative space composition, muted earth tone palette, geometric shapes, Scandinavian design, zen simplicity",
        "tags": "minimalist, geometric, Scandinavian, zen, clean",
        "is_active": True,
    },
    {
        "prompt_text": "retro 80s synthwave, purple sunset grid, chrome reflections, palm trees silhouette, cassette futurism, laser beams, nostalgic arcade",
        "tags": "synthwave, 80s, retro, chrome, arcade, palm trees",
        "is_active": True,
    },
    {
        "prompt_text": "dark fantasy horror, crimson mist, twisted gnarled trees, full moon, foggy graveyard, ominous atmosphere, gothic cathedral ruins",
        "tags": "dark fantasy, horror, gothic, mist, moon, graveyard",
        "is_active": True,
    },
]

# ── SubjectPrompt 种子数据（17条）──
SUBJECT_PROMPT_SEEDS = [
    {
        "prompt_text": "majestic dragon soaring through clouds, scales shimmering in sunlight, fantasy art, epic scale, dramatic composition",
        "tags": "dragon, fantasy, clouds, epic, majestic",
        "is_active": True,
    },
    {
        "prompt_text": "futuristic robot, sleek chrome design, LED accents, standing in urban environment, sci-fi concept art, highly detailed mechanical parts",
        "tags": "robot, futuristic, chrome, sci-fi, mechanical",
        "is_active": True,
    },
    {
        "prompt_text": "serene Japanese garden, cherry blossoms in full bloom, koi pond, stone lantern, zen atmosphere, soft morning light, tranquil scene",
        "tags": "Japanese garden, cherry blossoms, zen, tranquil, koi",
        "is_active": True,
    },
    {
        "prompt_text": "cosmic nebula, swirling galaxies, vibrant purple and blue colors, stars scattered, deep space, awe-inspiring astronomical photography",
        "tags": "nebula, galaxy, cosmic, space, stars, astronomical",
        "is_active": True,
    },
    {
        "prompt_text": "vintage steam locomotive, billowing smoke, golden hour lighting, rural countryside, nostalgic atmosphere, intricate mechanical details",
        "tags": "steam locomotive, vintage, smoke, countryside, nostalgic",
        "is_active": True,
    },
    # ── 新增主题提示词 ──
    {
        "prompt_text": "enchanted forest with bioluminescent mushrooms, ancient twisted oaks, fireflies dancing in twilight, moss-covered stones, magical atmosphere",
        "tags": "forest, bioluminescent, mushrooms, magical, twilight",
        "is_active": True,
    },
    {
        "prompt_text": "ancient temple ruins overgrown with ivy, crumbling stone pillars, hidden jungle, dappled sunlight, archaeological discovery, lost civilization",
        "tags": "temple, ruins, jungle, ancient, archaeological, civilization",
        "is_active": True,
    },
    {
        "prompt_text": "underwater city with glass domes and coral towers, schools of tropical fish, bioluminescent jellyfish, submarine light rays, Atlantis fantasy",
        "tags": "underwater, city, coral, jellyfish, Atlantis, fantasy",
        "is_active": True,
    },
    {
        "prompt_text": "medieval castle perched on a cliff at golden hour, banners fluttering in wind, moat with drawbridge, dramatic clouds, fairytale architecture",
        "tags": "castle, medieval, cliff, golden hour, fairytale, banners",
        "is_active": True,
    },
    {
        "prompt_text": "floating islands connected by rope bridges, waterfalls cascading into clouds, giant birds circling, fantasy sky realm, ethereal mist",
        "tags": "floating islands, waterfalls, sky realm, fantasy, mist",
        "is_active": True,
    },
    {
        "prompt_text": "crystal cave with rainbow light refractions, stalactites of amethyst and quartz, underground lake, mysterious glow, mineral treasures",
        "tags": "crystal cave, rainbow, amethyst, quartz, underground, glow",
        "is_active": True,
    },
    {
        "prompt_text": "samurai warrior standing in cherry blossom rain, gleaming katana, traditional armor, dramatic pose, petals swirling in wind, honor and strength",
        "tags": "samurai, cherry blossom, katana, armor, warrior, petals",
        "is_active": True,
    },
    {
        "prompt_text": "space station orbiting a ringed gas giant, solar panels glinting, distant stars, astronaut on spacewalk, sci-fi orbital architecture",
        "tags": "space station, gas giant, astronaut, orbital, sci-fi, stars",
        "is_active": True,
    },
    {
        "prompt_text": "desert oasis with palm trees and starlit sky, Bedouin tent, campfire glow, sand dunes, tranquil night, Milky Way overhead",
        "tags": "desert, oasis, palm trees, starlit, campfire, Milky Way",
        "is_active": True,
    },
    {
        "prompt_text": "bamboo forest with morning mist and sunbeams, stone pathway, traditional Chinese pavilion, serene landscape, zen meditation spot",
        "tags": "bamboo, mist, sunbeams, pavilion, Chinese, zen, landscape",
        "is_active": True,
    },
    {
        "prompt_text": "volcanic landscape with lava flows and obsidian cliffs, smoke plumes rising, dramatic orange glow against dark sky, otherworldly terrain",
        "tags": "volcanic, lava, obsidian, smoke, dramatic, otherworldly",
        "is_active": True,
    },
    {
        "prompt_text": "aurora borealis over a frozen lake, snow-covered pine forest, arctic wilderness, starry night, reflections on ice, ethereal green and violet ribbons",
        "tags": "aurora, frozen lake, arctic, pine forest, reflections, northern lights",
        "is_active": True,
    },
]

# ── QualityPrompt 种子数据（7条）──
QUALITY_PROMPT_SEEDS = [
    {
        "prompt_text": "masterpiece, best quality, ultra detailed, 8k resolution, sharp focus, professional lighting, award winning photography",
        "tags": "masterpiece, 8k, ultra detailed, professional, award winning",
        "is_active": True,
    },
    {
        "prompt_text": "highly detailed, intricate patterns, fine textures, crisp edges, photorealistic rendering, studio quality, professional grade",
        "tags": "highly detailed, intricate, photorealistic, studio quality",
        "is_active": True,
    },
    {
        "prompt_text": "cinematic lighting, dramatic shadows, film grain, anamorphic lens flare, Hollywood movie still, color graded, widescreen composition",
        "tags": "cinematic, dramatic, film grain, anamorphic, Hollywood",
        "is_active": True,
    },
    {
        "prompt_text": "volumetric lighting, god rays, atmospheric fog, depth of field, cinematic composition, moody atmosphere, professional cinematography",
        "tags": "volumetric, god rays, fog, depth of field, cinematic",
        "is_active": True,
    },
    {
        "prompt_text": "HDR photography, extreme dynamic range, vibrant colors, stunning contrast, natural lighting, vivid and lifelike, eye-catching composition",
        "tags": "HDR, dynamic range, vibrant, contrast, vivid",
        "is_active": True,
    },
    {
        "prompt_text": "trending on artstation, featured on Behance, professional digital art, concept art quality, polished and refined, visually stunning",
        "tags": "artstation, Behance, professional, concept art, polished",
        "is_active": True,
    },
    {
        "prompt_text": "hyperrealistic, 16k resolution, micro details, subsurface scattering, ray tracing, global illumination, physically based rendering",
        "tags": "hyperrealistic, 16k, micro details, ray tracing, PBR",
        "is_active": True,
    },
]

# ── AppSettings 种子数据 ──
APP_SETTINGS_SEEDS = [
    {
        "key": "inference.device",
        "value": "mps",  # macOS Metal Performance Shaders
    },
    {
        "key": "inference.batch_size",
        "value": "1",
    },
    {
        "key": "gui.theme",
        "value": "dark",
    },
    {
        "key": "gui.language",
        "value": "zh-CN",
    },
    {
        "key": "output.default_format",
        "value": "png",
    },
    {
        "key": "output.default_quality",
        "value": "95",
    },
]


# ═════════════════════════════════════════════════════════════
# 初始化函数
# ═════════════════════════════════════════════════════════════

def create_data_directory():
    """
    创建 data 目录（如果不存在）
    确保数据库文件有合适的存放位置
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"[INFO] 已创建数据目录: {DATA_DIR}")
    else:
        print(f"[INFO] 数据目录已存在: {DATA_DIR}")


def create_tables():
    """
    创建所有数据库表
    使用 peewee 的 create_tables 方法，安全建表（如果表已存在则跳过）
    """
    model_classes = [
        StreamDiffusionModel,
        DepthModel,
        LoraModel,
        PromptCategory,
        StylePrompt,
        SubjectPrompt,
        QualityPrompt,
        AppSettings,
    ]

    with database:
        database.create_tables(model_classes, safe=True)

    print(f"[INFO] 数据库表创建完成（共 {len(model_classes)} 个表）")
    for m in model_classes:
        print(f"  ✓ {m._meta.table_name}")


def seed_streamdiffusion_models():
    """插入 StreamDiffusionModel 种子数据"""
    count = 0
    for data in STREAMDIFFUSION_MODEL_SEEDS:
        _, created = StreamDiffusionModel.get_or_create(
            name=data["name"],
            defaults=data,
        )
        if created:
            count += 1
    print(f"[INFO] StreamDiffusionModel: 插入 {count} 条新记录，总计 {StreamDiffusionModel.select().count()} 条")


def seed_depth_models():
    """插入 DepthModel 种子数据"""
    count = 0
    for data in DEPTH_MODEL_SEEDS:
        _, created = DepthModel.get_or_create(
            name=data["name"],
            defaults=data,
        )
        if created:
            count += 1
    print(f"[INFO] DepthModel: 插入 {count} 条新记录，总计 {DepthModel.select().count()} 条")


def seed_lora_models():
    """插入 LoraModel 种子数据"""
    count = 0
    for data in LORA_MODEL_SEEDS:
        _, created = LoraModel.get_or_create(
            name=data["name"],
            defaults=data,
        )
        if created:
            count += 1
    print(f"[INFO] LoraModel: 插入 {count} 条新记录，总计 {LoraModel.select().count()} 条")


def seed_prompt_categories():
    """插入 PromptCategory 种子数据"""
    count = 0
    for data in PROMPT_CATEGORY_SEEDS:
        _, created = PromptCategory.get_or_create(
            name=data["name"],
            defaults=data,
        )
        if created:
            count += 1
    print(f"[INFO] PromptCategory: 插入 {count} 条新记录，总计 {PromptCategory.select().count()} 条")


def seed_style_prompts():
    """插入 StylePrompt 种子数据"""
    try:
        style_category = PromptCategory.get(PromptCategory.name == "style")
    except PromptCategory.DoesNotExist:
        print("[WARN] 跳过 StylePrompt：对应的 PromptCategory 'style' 不存在")
        return

    count = 0
    for data in STYLE_PROMPT_SEEDS:
        create_data = {
            "prompt_text": data["prompt_text"],
            "category": style_category,
            "tags": data.get("tags"),
            "is_active": data["is_active"],
        }
        _, created = StylePrompt.get_or_create(
            prompt_text=data["prompt_text"],
            defaults=create_data,
        )
        if created:
            count += 1
    print(f"[INFO] StylePrompt: 插入 {count} 条新记录，总计 {StylePrompt.select().count()} 条")


def seed_subject_prompts():
    """插入 SubjectPrompt 种子数据"""
    try:
        subject_category = PromptCategory.get(PromptCategory.name == "subject")
    except PromptCategory.DoesNotExist:
        print("[WARN] 跳过 SubjectPrompt：对应的 PromptCategory 'subject' 不存在")
        return

    count = 0
    for data in SUBJECT_PROMPT_SEEDS:
        create_data = {
            "prompt_text": data["prompt_text"],
            "category": subject_category,
            "tags": data.get("tags"),
            "is_active": data["is_active"],
        }
        _, created = SubjectPrompt.get_or_create(
            prompt_text=data["prompt_text"],
            defaults=create_data,
        )
        if created:
            count += 1
    print(f"[INFO] SubjectPrompt: 插入 {count} 条新记录，总计 {SubjectPrompt.select().count()} 条")


def seed_quality_prompts():
    """插入 QualityPrompt 种子数据"""
    try:
        quality_category = PromptCategory.get(PromptCategory.name == "quality")
    except PromptCategory.DoesNotExist:
        print("[WARN] 跳过 QualityPrompt：对应的 PromptCategory 'quality' 不存在")
        return

    count = 0
    for data in QUALITY_PROMPT_SEEDS:
        create_data = {
            "prompt_text": data["prompt_text"],
            "category": quality_category,
            "tags": data.get("tags"),
            "is_active": data["is_active"],
        }
        _, created = QualityPrompt.get_or_create(
            prompt_text=data["prompt_text"],
            defaults=create_data,
        )
        if created:
            count += 1
    print(f"[INFO] QualityPrompt: 插入 {count} 条新记录，总计 {QualityPrompt.select().count()} 条")


def seed_app_settings():
    """插入 AppSettings 种子数据"""
    count = 0
    for data in APP_SETTINGS_SEEDS:
        _, created = AppSettings.get_or_create(
            key=data["key"],
            defaults=data,
        )
        if created:
            count += 1
    print(f"[INFO] AppSettings: 插入 {count} 条新记录，总计 {AppSettings.select().count()} 条")


def verify_wal_mode():
    """验证数据库是否已启用 WAL 模式"""
    cursor = database.execute_sql("PRAGMA journal_mode;")
    result = cursor.fetchone()
    mode = result[0] if result else "unknown"
    print(f"[INFO] 数据库日志模式: {mode}")
    if mode.lower() == "wal":
        print("  ✓ WAL 模式已启用")
    else:
        print(f"  ⚠ 当前模式为 {mode}，建议启用 WAL 模式")


def init_db():
    """
    主初始化函数：执行完整的数据库初始化流程

    步骤：
        1. 创建 data 目录
        2. 连接数据库
        3. 创建所有表
        4. 插入种子数据
        5. 验证 WAL 模式
        6. 关闭连接
    """
    print("=" * 60)
    print("StreamDiffusion 数据库初始化")
    print("=" * 60)

    create_data_directory()
    database.connect(reuse_if_open=True)
    print(f"[INFO] 数据库连接: {DB_PATH}")

    create_tables()

    with database.atomic():
        seed_streamdiffusion_models()
        seed_depth_models()
        seed_lora_models()
        seed_prompt_categories()
        seed_style_prompts()
        seed_subject_prompts()
        seed_quality_prompts()
        seed_app_settings()

    verify_wal_mode()
    database.close()

    print("=" * 60)
    print("数据库初始化完成！")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# 脚本入口：直接运行此文件时执行初始化
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
