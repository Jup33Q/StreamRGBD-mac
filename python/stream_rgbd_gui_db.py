#!/usr/bin/env python3
"""
Stream RGBD GUI — 数据库集成增强版（按 Category 分组随机提示词）
为 stream_rgbd 提供可视化参数配置 + 启动/停止 + 日志输出 + 数据库管理。

基于 stream_rgbd_gui.py 扩展，集成 peewee 数据库模型：
- ModelType, Model, StylePrompt, SubjectPrompt, QualityPrompt, PromptCategory, AppSettings

提示词结构：style + subject + quality（按逗号拼接）
- style 可为空
- subject 不能为空（必填）
- quality 可为空

Usage:
    python stream_rgbd_gui_db.py
"""
import os
import sys
import json
import subprocess
import threading
import shlex
import random
import time

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
except ImportError:
    print("ERROR: tkinter 不可用。请确认 Python 安装了 Tk 支持。")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 尝试导入样式模块
# ---------------------------------------------------------------------------
try:
    from tk_style import GlassWindow, apply_glass_effect
except ImportError:
    GlassWindow = tk.Tk
    apply_glass_effect = lambda *a, **k: None

# ---------------------------------------------------------------------------
# 尝试导入数据库模型（带完整 fallback）
# ---------------------------------------------------------------------------
_db_available = False
try:
    from models import (
        db,
        StreamDiffusionModel, DepthModel, LoraModel,
        StyleLoraModel, SubjectLoraModel, QualityLoraModel,
        StylePrompt, SubjectPrompt, QualityPrompt,
        PromptCategory, AppSettings,
    )
    from db_init import init_db
    _db_available = True
except Exception as _e:
    print(f"[WARN] 数据库模型导入失败 ({_e})，将使用内置 fallback")
    db = None
    StreamDiffusionModel = None
    DepthModel = None
    LoraModel = None
    StyleLoraModel = None
    SubjectLoraModel = None
    QualityLoraModel = None
    StylePrompt = None
    SubjectPrompt = None
    QualityPrompt = None
    PromptCategory = None
    AppSettings = None
    init_db = None


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(PROJECT_DIR, "python", "camera_rgbd.py")
VENV_ACTIVATE = os.path.join(PROJECT_DIR, ".venv", "bin", "activate")

# ---------------------------------------------------------------------------
# 内置 fallback 数据（当数据库不可用时使用）
# ---------------------------------------------------------------------------
_FALLBACK_MODELS = [
    ("sdxs", "基础模型"),
    ("sd-turbo", "基础模型"),
]

_FALLBACK_DEPTH_MODELS = [
    ("auto", "自动选择"),
    ("da3-small", "Depth Anything V3 Small"),
    ("da2-small", "Depth Anything V2 Small"),
]

# 按 category 拆分的 fallback 提示词
_FALLBACK_STYLE_PROMPTS = [
    ("oil painting, classical portrait", "油画"),
    ("watercolor painting, soft pastel colors", "水彩"),
    ("cyberpunk cityscape, neon lights", "赛博朋克"),
    ("anime style, vibrant colors, cel shading", "动漫"),
    ("photorealistic portrait, natural lighting", "写实"),
]

_FALLBACK_SUBJECT_PROMPTS = [
    ("majestic dragon soaring through clouds", "龙"),
    ("futuristic robot, sleek chrome design", "机器人"),
    ("serene Japanese garden, cherry blossoms", "日式庭院"),
    ("cosmic nebula, swirling galaxies", "星云"),
    ("vintage steam locomotive, billowing smoke", "蒸汽机车"),
]

_FALLBACK_QUALITY_PROMPTS = [
    ("masterpiece, best quality, ultra detailed, 8k", "8K"),
    ("cinematic lighting, dramatic shadows, film grain", "电影级"),
    ("volumetric lighting, god rays, atmospheric fog", "体积光"),
    ("HDR photography, extreme dynamic range", "HDR"),
    ("trending on artstation, professional digital art", "ArtStation"),
]

# 按 category 拆分的 fallback LoRA
# 每项: (name, display_name, file_path, weight_min, weight_max, weight_default)
_FALLBACK_LORAS = {
    "style": [
        ("pixelart-redmond", "PixelArt 像素艺术", "loras/pixelart_redmond.safetensors", 0.6, 1.0, 0.8),
        ("sketch-sd15", "Sketch 素描", "loras/sketch_sd15.safetensors", 0.5, 0.9, 0.7),
        ("civitai-moxin-ink", "MoXin 中国水墨", "loras/moxin-ink.safetensors", 0.6, 1.0, 0.8),
    ],
    "subject": [
        ("victorian-dress", "Victorian Dress 维多利亚礼服", "loras/victorian_dress.safetensors", 0.5, 1.0, 0.75),
        ("kimono-outfit", "Kimono Outfit 和服", "loras/kimono_outfit.safetensors", 0.5, 1.0, 0.75),
        ("cyberpunk-outfit", "Cyberpunk Outfit 赛博朋克服装", "loras/cyberpunk_outfit.safetensors", 0.5, 1.0, 0.75),
    ],
    "quality": [
        ("civitai-detail-tweaker", "Detail Tweaker 细节增强", "loras/detail-tweaker.safetensors", -2.0, 2.0, 0.5),
        ("civitai-more-details", "More Details 更多细节", "loras/more-details.safetensors", 0.3, 1.0, 0.6),
    ],
}


class StreamRGBDGUIDB:
    """Stream RGBD 控制台 — 数据库增强版（按 Category 分组提示词）。"""

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def __init__(self):
        self._proc = None
        self._reader_thread = None
        self._running = False
        self._db_available = _db_available
        self._toast_window = None

        # 初始化数据库（如果可用）
        if self._db_available and init_db is not None:
            try:
                init_db()
                print("[INFO] 数据库初始化成功")
            except Exception as e:
                print(f"[WARN] 数据库初始化失败: {e}")
                self._db_available = False

        self.root = GlassWindow(material='popover', dark=False, overlay=False)
        self.root.title("Stream RGBD 控制台 (数据库版)")
        self.root.geometry("920x780")
        self.root.minsize(720, 580)

        # ========== 左侧：可滚动参数面板 ==========
        left_container = ttk.Frame(self.root, padding="0")
        left_container.pack(side=tk.LEFT, fill=tk.Y)

        # --- 顶部工具栏（控制按钮，置顶不滚动）---
        toolbar = ttk.Frame(left_container, padding="10 8")
        toolbar.pack(fill=tk.X, side=tk.TOP)

        self.btn_start = ttk.Button(toolbar, text="▶ 启动", command=self._on_start)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_stop = ttk.Button(toolbar, text="⏹ 停止", command=self._on_stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_clear = ttk.Button(toolbar, text="清空日志", command=self._on_clear_log)
        self.btn_clear.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_status = ttk.Label(toolbar, text="状态: 就绪", foreground="gray")
        self.lbl_status.pack(side=tk.RIGHT)

        canvas = tk.Canvas(left_container, width=360, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        left = ttk.Frame(canvas, padding="10")
        canvas_window = canvas.create_window((0, 0), window=left, anchor=tk.NW)

        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=event.width)
        left.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))

        # 鼠标滚轮滚动
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ------------------------------------------------------------------
        # --- Prompt 按 Category 分组（核心重构）---
        # ------------------------------------------------------------------
        ttk.Label(left, text="提示词构建（style + subject + quality）", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, pady=(0, 6))

        # --- Style 行 ---
        style_frame = ttk.Frame(left)
        style_frame.pack(fill=tk.X, pady=(0, 2))
        self.btn_random_style = tk.Button(
            style_frame, text="🎲 Style", command=lambda: self._on_random_category("style"),
            bg="#9C27B0", fg="white", activebackground="#7B1FA2", activeforeground="white",
            font=("Helvetica", 9, "bold"), cursor="hand2", width=8,
        )
        self.btn_random_style.pack(side=tk.LEFT, padx=(0, 4))
        self.entry_style = ttk.Entry(style_frame, width=24)
        self.entry_style.insert(0, "oil painting, classical portrait")
        self.entry_style.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry_style.bind("<KeyRelease>", lambda e: self._update_combined_prompt())

        # --- Style LoRA 行 ---
        self._make_lora_row(left, "style")

        # --- Subject 行（必填）---
        subject_frame = ttk.Frame(left)
        subject_frame.pack(fill=tk.X, pady=(0, 2))
        self.btn_random_subject = tk.Button(
            subject_frame, text="🎲 Subject", command=lambda: self._on_random_category("subject"),
            bg="#E91E63", fg="white", activebackground="#C2185B", activeforeground="white",
            font=("Helvetica", 9, "bold"), cursor="hand2", width=8,
        )
        self.btn_random_subject.pack(side=tk.LEFT, padx=(0, 4))
        self.entry_subject = ttk.Entry(subject_frame, width=24)
        self.entry_subject.insert(0, "majestic dragon soaring through clouds")
        self.entry_subject.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry_subject.bind("<KeyRelease>", lambda e: self._update_combined_prompt())

        # --- Subject LoRA 行 ---
        self._make_lora_row(left, "subject")

        # --- Quality 行 ---
        quality_frame = ttk.Frame(left)
        quality_frame.pack(fill=tk.X, pady=(0, 2))
        self.btn_random_quality = tk.Button(
            quality_frame, text="🎲 Quality", command=lambda: self._on_random_category("quality"),
            bg="#3F51B5", fg="white", activebackground="#303F9F", activeforeground="white",
            font=("Helvetica", 9, "bold"), cursor="hand2", width=8,
        )
        self.btn_random_quality.pack(side=tk.LEFT, padx=(0, 4))
        self.entry_quality = ttk.Entry(quality_frame, width=24)
        self.entry_quality.insert(0, "masterpiece, best quality, ultra detailed, 8k")
        self.entry_quality.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry_quality.bind("<KeyRelease>", lambda e: self._update_combined_prompt())

        # --- Quality LoRA 行 ---
        self._make_lora_row(left, "quality")

        # --- 合并提示词显示（可复制，带复制按钮）---
        combined_frame = ttk.LabelFrame(left, text="合并提示词（自动拼接，可拖拽选中复制）", padding="5")
        combined_frame.pack(fill=tk.X, pady=(6, 4))
        combined_inner = ttk.Frame(combined_frame)
        combined_inner.pack(fill=tk.X)
        self.entry_combined = tk.Entry(combined_inner, width=38, state="readonly",
                                       readonlybackground="#f5f5f5", fg="#333333",
                                       font=("Menlo", 10), selectbackground="#B39DDB")
        self.entry_combined.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.btn_copy_combined = tk.Button(
            combined_inner, text="📋", command=self._copy_combined_prompt,
            bg="#4CAF50", fg="white", activebackground="#45a049", activeforeground="white",
            font=("Helvetica", 9, "bold"), cursor="hand2", width=3,
        )
        self.btn_copy_combined.pack(side=tk.LEFT)

        # --- 全部随机按钮 ---
        all_random_frame = ttk.Frame(left)
        all_random_frame.pack(fill=tk.X, pady=(0, 8))
        self.btn_random_all = tk.Button(
            all_random_frame, text="🎲🎲🎲 全部随机", command=self._on_random_all,
            bg="#FF5722", fg="white", activebackground="#E64A19", activeforeground="white",
            font=("Helvetica", 10, "bold"), cursor="hand2",
        )
        self.btn_random_all.pack(side=tk.LEFT, padx=(0, 8))
        self.lbl_combined_preview = ttk.Label(all_random_frame, text="", foreground="gray", wraplength=280)
        self.lbl_combined_preview.pack(side=tk.LEFT, anchor=tk.W)

        # 初始化合并提示词
        self._update_combined_prompt()

        # ------------------------------------------------------------------
        # --- Runtime Prompt Update（多行 + 粘贴替换）---
        # ------------------------------------------------------------------
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(left, text="新提示词 (运行中可更新):").pack(anchor=tk.W, pady=(0, 2))
        self.entry_new_prompt = tk.Text(
            left, width=40, height=3, wrap=tk.WORD,
            font=("Menlo", 10), bg="#fafafa", fg="#333333",
            insertbackground="#333333", padx=4, pady=4,
        )
        self.entry_new_prompt.insert("1.0", "watercolor painting, soft brushstrokes")
        self.entry_new_prompt.pack(fill=tk.X, pady=(0, 4))

        # 按钮栏：更新 + 粘贴替换
        prompt_frame = ttk.Frame(left)
        prompt_frame.pack(fill=tk.X, pady=(0, 8))
        self.btn_update_prompt = tk.Button(
            prompt_frame, text="更新提示词", command=self._on_update_prompt,
            bg="#4CAF50", fg="black", activebackground="#45a049", activeforeground="black",
            font=("Helvetica", 11, "bold"), cursor="hand2", state=tk.DISABLED,
        )
        self.btn_update_prompt.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_paste_prompt = tk.Button(
            prompt_frame, text="📋 粘贴替换", command=self._paste_new_prompt,
            bg="#2196F3", fg="white", activebackground="#1976D2", activeforeground="white",
            font=("Helvetica", 10, "bold"), cursor="hand2",
        )
        self.btn_paste_prompt.pack(side=tk.LEFT, padx=(0, 8))
        self.lbl_current_prompt = ttk.Label(prompt_frame, text="当前: (未运行)", foreground="gray", wraplength=280)
        self.lbl_current_prompt.pack(side=tk.LEFT, anchor=tk.W)

        # ------------------------------------------------------------------
        # --- Seed ---
        # ------------------------------------------------------------------
        ttk.Label(left, text="Seed (随机种子):").pack(anchor=tk.W, pady=(0, 2))
        seed_frame = ttk.Frame(left)
        seed_frame.pack(fill=tk.X, pady=(0, 8))
        self.entry_seed = ttk.Entry(seed_frame, width=15)
        self.entry_seed.insert(0, "42")
        self.entry_seed.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_random_seed = tk.Button(
            seed_frame, text="🎲 随机", command=self._on_random_seed,
            bg="#FF9800", fg="black", activebackground="#F57C00", activeforeground="black",
            font=("Helvetica", 10, "bold"), cursor="hand2",
        )
        self.btn_random_seed.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_update_seed = tk.Button(
            seed_frame, text="应用 Seed", command=self._on_update_seed,
            bg="#2196F3", fg="black", activebackground="#1976D2", activeforeground="black",
            font=("Helvetica", 10, "bold"), cursor="hand2", state=tk.DISABLED,
        )
        self.btn_update_seed.pack(side=tk.LEFT)
        self.lbl_current_seed = ttk.Label(left, text="当前 Seed: 42", foreground="gray")
        self.lbl_current_seed.pack(anchor=tk.W, pady=(0, 4))

        # ------------------------------------------------------------------
        # --- Model（从数据库加载）---
        # ------------------------------------------------------------------
        ttk.Label(left, text="Model:").pack(anchor=tk.W, pady=(0, 2))
        self.combo_model = ttk.Combobox(left, state="readonly", width=20)
        self._load_models()
        self.combo_model.pack(anchor=tk.W, pady=(0, 8))

        # ------------------------------------------------------------------
        # --- Output Resolution ---
        # ------------------------------------------------------------------
        ttk.Label(left, text="Output Resolution:").pack(anchor=tk.W, pady=(0, 2))
        self.combo_render = ttk.Combobox(
            left,
            values=["512x512", "768x768", "384x384", "320x320", "720x1280"],
            state="readonly",
            width=20,
        )
        self.combo_render.set("512x512")
        self.combo_render.pack(anchor=tk.W, pady=(0, 8))

        # ------------------------------------------------------------------
        # --- Depth ---
        # ------------------------------------------------------------------
        ttk.Label(left, text="Depth Backend:").pack(anchor=tk.W, pady=(0, 2))
        self.combo_depth = ttk.Combobox(left, values=["auto", "coreml", "pytorch"], state="readonly", width=20)
        self.combo_depth.set("auto")
        self.combo_depth.pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(left, text="Depth Model:").pack(anchor=tk.W, pady=(0, 2))
        self.combo_depth_model = ttk.Combobox(left, state="readonly", width=20)
        self._load_depth_models()
        self.combo_depth_model.pack(anchor=tk.W, pady=(0, 8))

        # ------------------------------------------------------------------
        # --- NDI ---
        # ------------------------------------------------------------------
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(left, text="NDI 输出名称:").pack(anchor=tk.W, pady=(0, 2))
        self.entry_ndi = ttk.Entry(left, width=45)
        self.entry_ndi.insert(0, "StreamDiffusion-RGBD")
        self.entry_ndi.pack(fill=tk.X, pady=(0, 8))

        # ------------------------------------------------------------------
        # --- Strength / Blend / EMA (Slider + Value) ---
        # ------------------------------------------------------------------
        def _make_slider(parent, label, from_, to, resolution, default, fmt):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=(0, 4))
            ttk.Label(frame, text=label).pack(side=tk.LEFT)
            slider = tk.Scale(
                frame, from_=from_, to=to, orient=tk.HORIZONTAL,
                resolution=resolution, showvalue=False,
            )
            slider.set(default)
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
            lbl = ttk.Label(frame, text=fmt.format(default), width=6)
            lbl.pack(side=tk.LEFT)
            slider.config(command=lambda v, l=lbl, f=fmt: l.config(text=f.format(float(v))))
            return slider

        self.slider_strength = _make_slider(left, "Strength:", 0.0, 1.0, 0.01, 0.5, "{:.2f}")
        self.slider_blend = _make_slider(left, "Blend:", 0.0, 1.0, 0.01, 0.0, "{:.2f}")
        self.slider_ema = _make_slider(left, "EMA:", 0.0, 0.99, 0.01, 0.4, "{:.2f}")

        # ------------------------------------------------------------------
        # --- Extra Args ---
        # ------------------------------------------------------------------
        ttk.Label(left, text="额外参数:").pack(anchor=tk.W, pady=(0, 2))
        self.entry_extra = ttk.Entry(left, width=45)
        self.entry_extra.pack(fill=tk.X, pady=(0, 8))

        # ------------------------------------------------------------------
        # --- 数据库管理按钮 ---
        # ------------------------------------------------------------------
        db_btn_frame = ttk.Frame(left)
        db_btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.btn_model_mgr = tk.Button(
            db_btn_frame, text="📂 模型管理", command=self._open_model_manager,
            bg="#607D8B", fg="white", activebackground="#455A64", activeforeground="white",
            font=("Helvetica", 10, "bold"), cursor="hand2",
        )
        self.btn_model_mgr.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_prompt_mgr = tk.Button(
            db_btn_frame, text="📝 提示词管理", command=self._open_prompt_manager,
            bg="#607D8B", fg="white", activebackground="#455A64", activeforeground="white",
            font=("Helvetica", 10, "bold"), cursor="hand2",
        )
        self.btn_prompt_mgr.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_save_settings = tk.Button(
            db_btn_frame, text="💾 保存设置", command=self._save_settings,
            bg="#009688", fg="white", activebackground="#00796B", activeforeground="white",
            font=("Helvetica", 10, "bold"), cursor="hand2",
        )
        self.btn_save_settings.pack(side=tk.LEFT)

        # 数据库状态标签
        db_status_text = "数据库: 已连接" if self._db_available else "数据库: 离线 (fallback)"
        db_status_color = "green" if self._db_available else "orange"
        self.lbl_db_status = ttk.Label(left, text=db_status_text, foreground=db_status_color)
        self.lbl_db_status.pack(anchor=tk.W, pady=(0, 4))



        # ========== 右侧：日志输出 ==========
        right = ttk.Frame(self.root, padding="10")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(right, text="日志输出:").pack(anchor=tk.W, pady=(0, 5))
        self.txt_log = scrolledtext.ScrolledText(right, wrap=tk.WORD, state=tk.DISABLED, font=("Menlo", 11))
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 启动时尝试加载保存的设置
        self._load_settings()

    # ------------------------------------------------------------------
    # 合并提示词：自动拼接 style + subject + quality
    # ------------------------------------------------------------------
    def _update_combined_prompt(self):
        """将 style、subject、quality 按逗号拼接为合并提示词，并更新显示。"""
        style = self.entry_style.get().strip()
        subject = self.entry_subject.get().strip()
        quality = self.entry_quality.get().strip()

        parts = []
        if style:
            parts.append(style)
        if subject:
            parts.append(subject)
        if quality:
            parts.append(quality)

        combined = ", ".join(parts) if parts else ""

        # 更新只读显示框
        self.entry_combined.config(state=tk.NORMAL)
        self.entry_combined.delete(0, tk.END)
        self.entry_combined.insert(0, combined)
        self.entry_combined.config(state="readonly")

        # 更新预览标签
        preview = combined[:50] + "..." if len(combined) > 50 else combined
        self.lbl_combined_preview.config(text=preview)

        return combined

    def _copy_combined_prompt(self):
        """将合并提示词复制到系统剪贴板，并显示气泡提示。"""
        combined = self._update_combined_prompt()
        self.root.clipboard_clear()
        self.root.clipboard_append(combined)
        self.root.update()  # 确保剪贴板更新
        self._log(f"[COPY] 已复制到剪贴板: {combined[:50]}...")
        # 短暂闪烁按钮颜色作为反馈
        self.btn_copy_combined.config(bg="#2E7D32")
        self.root.after(300, lambda: self.btn_copy_combined.config(bg="#4CAF50"))

    # ------------------------------------------------------------------
    # 数据库相关：加载模型列表
    # ------------------------------------------------------------------
    def _load_models(self):
        """从 StreamDiffusionModel 表加载 SD 模型列表。"""
        if self._db_available and StreamDiffusionModel is not None:
            try:
                models = StreamDiffusionModel.select().where(StreamDiffusionModel.is_active == True)
                values = []
                for m in models:
                    kind_label = "T2I" if m.model_kind == "t2i" else "Img2Img"
                    values.append(f"{m.name} ({kind_label})")
                if values:
                    self.combo_model["values"] = values
                    self.combo_model.set(values[0])
                    return
            except Exception as e:
                self._log(f"[WARN] 加载模型列表失败: {e}")
        # Fallback
        values = [f"{name} ({desc})" for name, desc in _FALLBACK_MODELS]
        self.combo_model["values"] = values
        self.combo_model.set(values[0])

    def _load_depth_models(self):
        """从 DepthModel 表加载深度模型列表。"""
        if self._db_available and DepthModel is not None:
            try:
                models = DepthModel.select().where(DepthModel.is_active == True)
                values = [f"{m.name} ({m.display_name})" for m in models]
                if values:
                    self.combo_depth_model["values"] = values
                    self.combo_depth_model.set(values[0])
                    return
            except Exception as e:
                self._log(f"[WARN] 加载深度模型列表失败: {e}")
        # Fallback
        values = [f"{name} ({desc})" for name, desc in _FALLBACK_DEPTH_MODELS]
        self.combo_depth_model["values"] = values
        self.combo_depth_model.set(values[0])

    # ------------------------------------------------------------------
    # 按 Category 加载 LoRA（拆表 + slider 权重）
    # ------------------------------------------------------------------
    def _load_lora_data(self):
        """从分类 LoRA 表加载数据，返回 {category: [dict, ...]}。"""
        data = {"style": [], "subject": [], "quality": []}
        if self._db_available:
            table_map = {
                "style": (StyleLoraModel, "style_subtype"),
                "subject": (SubjectLoraModel, "subject_subtype"),
                "quality": (QualityLoraModel, "quality_subtype"),
            }
            for category, (table, subtype_field) in table_map.items():
                try:
                    for m in table.select().where(table.is_active == True):
                        data[category].append({
                            "name": m.name,
                            "display_name": m.display_name or m.name,
                            "path": m.file_path,
                            "weight_min": m.weight_min,
                            "weight_max": m.weight_max,
                            "weight_default": m.weight_default,
                            "sub_type": getattr(m, subtype_field, "other"),
                        })
                except Exception as e:
                    self._log(f"[WARN] 加载 {category} LoRA 失败: {e}")
        # Fallback
        for category, items in _FALLBACK_LORAS.items():
            if not data[category]:
                for name, display_name, path, wmin, wmax, wdefault in items:
                    data[category].append({
                        "name": name,
                        "display_name": display_name,
                        "path": path,
                        "weight_min": wmin,
                        "weight_max": wmax,
                        "weight_default": wdefault,
                        "sub_type": "other",
                    })
        return data

    def _resolve_lora_path(self, path):
        """将 LoRA 相对路径解析为实际存在的绝对路径。"""
        if not path:
            return path
        if os.path.isabs(path) and os.path.exists(path):
            return path
        # 尝试项目根目录
        for base in [PROJECT_DIR, os.path.dirname(os.path.abspath(__file__))]:
            candidate = os.path.join(base, path)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        # 尝试 python/loras/ 目录（按文件名匹配）
        loras_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loras")
        basename = os.path.basename(path)
        candidate = os.path.join(loras_dir, basename)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        return path

    def _make_lora_row(self, parent, category: str):
        """为指定 category 创建 LoRA 选择器 + 权重滑块行。"""
        if not hasattr(self, "_lora_data"):
            self._lora_data = self._load_lora_data()
        if not hasattr(self, "_lora_widgets"):
            self._lora_widgets = {}
        if not hasattr(self, "_lora_selection"):
            self._lora_selection = {}
        if not hasattr(self, "_lora_debounce_id"):
            self._lora_debounce_id = None

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(frame, text=f"{category.capitalize()} LoRA:", width=12).pack(side=tk.LEFT)

        choices = [("None", None)]
        for item in self._lora_data.get(category, []):
            label = f"{item['display_name']} ({item['name']})"
            choices.append((label, item))

        combo = ttk.Combobox(frame, state="readonly", width=22)
        combo["values"] = [c[0] for c in choices]
        combo.set(choices[0][0])
        combo.pack(side=tk.LEFT, padx=(0, 6))

        # Default weight limits until a LoRA is selected
        wmin, wmax, wdefault = -2.0, 2.0, 0.0
        slider = tk.Scale(
            frame, from_=wmin, to=wmax, orient=tk.HORIZONTAL,
            resolution=0.05, showvalue=False, length=120,
        )
        slider.set(wdefault)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        lbl = ttk.Label(frame, text=f"{wdefault:.2f}", width=6)
        lbl.pack(side=tk.LEFT)

        self._lora_widgets[category] = {
            "frame": frame,
            "combo": combo,
            "slider": slider,
            "label": lbl,
            "choices": choices,
        }
        self._lora_selection[category] = {"item": None, "weight": 0.0}

        def _on_combo_change(evt, cat=category):
            self._on_lora_selection_changed(cat)

        def _on_slider_change(value, cat=category):
            self._on_lora_slider_changed(cat, float(value))

        combo.bind("<<ComboboxSelected>>", _on_combo_change)
        slider.config(command=_on_slider_change)

    def _on_lora_selection_changed(self, category: str):
        """LoRA 选择变化时更新 slider 的限位和默认值。"""
        widgets = self._lora_widgets[category]
        combo = widgets["combo"]
        slider = widgets["slider"]
        lbl = widgets["label"]
        selected_label = combo.get()

        item = None
        for label, it in widgets["choices"]:
            if label == selected_label:
                item = it
                break

        if item is None:
            # None selected
            slider.config(from_=-2.0, to=2.0)
            slider.set(0.0)
            lbl.config(text="0.00")
            self._lora_selection[category] = {"item": None, "weight": 0.0}
        else:
            wmin, wmax, wdefault = item["weight_min"], item["weight_max"], item["weight_default"]
            slider.config(from_=wmin, to=wmax)
            slider.set(wdefault)
            lbl.config(text=f"{wdefault:.2f}")
            self._lora_selection[category] = {"item": item, "weight": wdefault}

        self._on_lora_update()

    def _on_lora_slider_changed(self, category: str, value: float):
        """Slider 数值变化时更新显示并触发 debounced 更新。"""
        widgets = self._lora_widgets[category]
        widgets["label"].config(text=f"{value:.2f}")
        sel = self._lora_selection.get(category, {})
        if sel.get("item") is not None:
            sel["weight"] = value
        self._on_lora_update()

    def _on_lora_update(self):
        """Debounced: 构建 LoRA stack 并发送到运行中的子进程。"""
        if self._lora_debounce_id is not None:
            self.root.after_cancel(self._lora_debounce_id)
        self._lora_debounce_id = self.root.after(250, self._send_lora_stack)

    def _get_lora_stack(self):
        """根据当前 UI 选择构建 LoRA stack（启动参数用）。"""
        stack = []
        for category in ["style", "subject", "quality"]:
            sel = self._lora_selection.get(category, {})
            item = sel.get("item")
            weight = sel.get("weight", 0.0)
            if item is not None and weight != 0.0:
                stack.append({
                    "path": self._resolve_lora_path(item["path"]),
                    "weight": weight,
                    "category": category,
                })
        return stack

    def _send_lora_stack(self):
        """向运行中的子进程发送 LoRA stack 更新。"""
        self._lora_debounce_id = None
        stack = self._get_lora_stack()
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            payload = json.dumps(stack, ensure_ascii=False)
            self._proc.stdin.write(f"lora:{payload}\n")
            self._proc.stdin.flush()
            self._log(f"[LORA] 已发送 LoRA stack: {len(stack)} 个")
        except Exception as e:
            self._log(f"[ERROR] 发送 LoRA stack 失败: {e}")

    # ------------------------------------------------------------------
    # 按 Category 随机提示词（核心重构）
    # ------------------------------------------------------------------
    def _on_random_category(self, category: str):
        """从指定 category 随机选取一条提示词填入对应输入框。"""
        prompt_text = None

        if self._db_available:
            table_map = {
                "style": StylePrompt,
                "subject": SubjectPrompt,
                "quality": QualityPrompt,
            }
            table = table_map.get(category)
            if table is not None:
                try:
                    active = list(table.select().where(table.is_active == True))
                    if active:
                        chosen = random.choice(active)
                        prompt_text = chosen.prompt_text
                        chosen.usage_count = (chosen.usage_count or 0) + 1
                        chosen.save()
                    else:
                        self._log(f"[INFO] 数据库中 category='{category}' 没有活跃提示词，使用 fallback")
                except Exception as e:
                    self._log(f"[WARN] 数据库查询 {category} 失败: {e}")

        # Fallback
        if prompt_text is None:
            fallback_map = {
                "style": _FALLBACK_STYLE_PROMPTS,
                "subject": _FALLBACK_SUBJECT_PROMPTS,
                "quality": _FALLBACK_QUALITY_PROMPTS,
            }
            fallback_list = fallback_map.get(category, [])
            if fallback_list:
                chosen = random.choice(fallback_list)
                prompt_text = chosen[0]

        # 填入对应输入框
        if category == "style":
            self.entry_style.delete(0, tk.END)
            self.entry_style.insert(0, prompt_text)
        elif category == "subject":
            self.entry_subject.delete(0, tk.END)
            self.entry_subject.insert(0, prompt_text)
        elif category == "quality":
            self.entry_quality.delete(0, tk.END)
            self.entry_quality.insert(0, prompt_text)

        # 自动更新合并提示词
        self._update_combined_prompt()

        # 气泡提示
        self._show_prompt_toast(category, prompt_text)
        self._log(f"[PROMPT] {category}: {prompt_text[:50]}...")

    def _on_random_all(self):
        """全部随机：同时随机 style、subject、quality，并更新合并提示词。"""
        for category in ["style", "subject", "quality"]:
            self._on_random_category(category)

    def _show_prompt_toast(self, category: str, prompt_text: str):
        """显示一个短暂的气泡提示窗口。"""
        # 关闭旧的气泡
        if self._toast_window is not None and self._toast_window.winfo_exists():
            self._toast_window.destroy()

        # 根据 category 选择颜色
        colors = {
            "style": ("#9C27B0", "#E1BEE7"),      # 紫色
            "subject": ("#E91E63", "#F8BBD9"),   # 粉红
            "quality": ("#3F51B5", "#C5CAE9"),   # 靛蓝
        }
        bg, fg = colors.get(category, ("#9C27B0", "#E1BEE7"))

        toast = tk.Toplevel(self.root)
        self._toast_window = toast
        toast.overrideredirect(True)
        toast.attributes("-topmost", True)
        toast.attributes("-alpha", 0.0)

        frame = tk.Frame(toast, bg=bg, padx=16, pady=10)
        frame.pack()

        lbl_category = tk.Label(
            frame, text=f"📂 {category.upper()}",
            bg=bg, fg="white",
            font=("Helvetica", 11, "bold")
        )
        lbl_category.pack(anchor=tk.W)

        preview = prompt_text[:60] + "..." if len(prompt_text) > 60 else prompt_text
        lbl_preview = tk.Label(
            frame, text=preview,
            bg=bg, fg=fg,
            font=("Helvetica", 10), wraplength=280, justify=tk.LEFT
        )
        lbl_preview.pack(anchor=tk.W, pady=(4, 0))

        self.root.update_idletasks()
        x = self.root.winfo_x() + 20
        y = self.root.winfo_y() + 80
        toast.geometry(f"+{x}+{y}")

        def _fade_in(alpha=0.0):
            if alpha < 0.92:
                toast.attributes("-alpha", alpha)
                self.root.after(20, lambda: _fade_in(alpha + 0.08))
            else:
                toast.attributes("-alpha", 0.92)
        _fade_in()

        def _close_toast():
            if toast.winfo_exists():
                def _fade_out(alpha=0.92):
                    if alpha > 0:
                        toast.attributes("-alpha", alpha)
                        self.root.after(20, lambda: _fade_out(alpha - 0.08))
                    else:
                        toast.destroy()
                _fade_out()
        self.root.after(2500, _close_toast)

    # ------------------------------------------------------------------
    # 模型管理窗口
    # ------------------------------------------------------------------
    def _open_model_manager(self):
        """打开模型列表管理窗口（五表分 Tab 展示）。"""
        win = tk.Toplevel(self.root)
        win.title("模型管理")
        win.geometry("800x520")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="已注册模型列表", font=("Helvetica", 14, "bold")).pack(pady=10)

        notebook = ttk.Notebook(win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # (tab_name, table_class, subtype_field, fallback_list)
        tabs_config = [
            ("StreamDiffusion", StreamDiffusionModel, None, _FALLBACK_MODELS),
            ("Depth", DepthModel, None, _FALLBACK_DEPTH_MODELS),
            ("Style-LoRA", StyleLoraModel, "style_subtype", []),
            ("Subject-LoRA", SubjectLoraModel, "subject_subtype", []),
            ("Quality-LoRA", QualityLoraModel, "quality_subtype", []),
        ]

        for tab_name, table_cls, subtype_field, fallback_list in tabs_config:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=tab_name)

            is_lora = subtype_field is not None
            if is_lora:
                columns = ("name", "display", "subtype", "weight", "path", "active")
                tree = ttk.Treeview(frame, columns=columns, show="headings", height=14)
                tree.heading("name", text="模型名")
                tree.heading("display", text="显示名")
                tree.heading("subtype", text="二级分类")
                tree.heading("weight", text="权重范围")
                tree.heading("path", text="路径")
                tree.heading("active", text="启用")
                tree.column("name", width=120)
                tree.column("display", width=150)
                tree.column("subtype", width=80, anchor=tk.CENTER)
                tree.column("weight", width=110, anchor=tk.CENTER)
                tree.column("path", width=180)
                tree.column("active", width=50, anchor=tk.CENTER)
            else:
                columns = ("name", "display", "path", "active")
                tree = ttk.Treeview(frame, columns=columns, show="headings", height=14)
                tree.heading("name", text="模型名")
                tree.heading("display", text="显示名")
                tree.heading("path", text="路径")
                tree.heading("active", text="启用")
                tree.column("name", width=120)
                tree.column("display", width=200)
                tree.column("path", width=300)
                tree.column("active", width=50, anchor=tk.CENTER)

            scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            if self._db_available and table_cls is not None:
                try:
                    for m in table_cls.select():
                        if is_lora:
                            weight_str = f"{m.weight_min}~{m.weight_max} (d={m.weight_default})"
                            subtype = getattr(m, subtype_field, "other") or "other"
                            tree.insert("", tk.END, values=(
                                m.name,
                                m.display_name or "—",
                                subtype,
                                weight_str,
                                getattr(m, "file_path", "—") or "—",
                                "✓" if m.is_active else "✗"
                            ))
                        else:
                            tree.insert("", tk.END, values=(
                                m.name,
                                m.display_name or "—",
                                getattr(m, "file_path", "—") or "—",
                                "✓" if m.is_active else "✗"
                            ))
                except Exception as e:
                    messagebox.showerror("错误", f"加载 {tab_name} 模型数据失败: {e}", parent=win)
            else:
                if fallback_list:
                    for name, desc in fallback_list:
                        tree.insert("", tk.END, values=(name, desc, "—", "✓"))
                elif is_lora:
                    tree.insert("", tk.END, values=("数据库未连接", "—", "—", "—", "—", "—"))
                else:
                    tree.insert("", tk.END, values=("数据库未连接", "—", "—", "—"))

        ttk.Button(win, text="关闭", command=win.destroy).pack(pady=10)

    # ------------------------------------------------------------------
    # 提示词管理窗口
    # ------------------------------------------------------------------
    def _open_prompt_manager(self):
        """打开提示词列表管理窗口（三表分 Tab 展示）。"""
        win = tk.Toplevel(self.root)
        win.title("提示词管理")
        win.geometry("750x500")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="默认提示词列表", font=("Helvetica", 14, "bold")).pack(pady=10)

        # Notebook 三表分 Tab
        notebook = ttk.Notebook(win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        tabs_config = [
            ("Style", StylePrompt, _FALLBACK_STYLE_PROMPTS),
            ("Subject", SubjectPrompt, _FALLBACK_SUBJECT_PROMPTS),
            ("Quality", QualityPrompt, _FALLBACK_QUALITY_PROMPTS),
        ]

        for tab_name, table_cls, fallback_list in tabs_config:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=tab_name)

            columns = ("prompt", "usage", "active")
            tree = ttk.Treeview(frame, columns=columns, show="headings", height=14)
            tree.heading("prompt", text="提示词")
            tree.heading("usage", text="使用次数")
            tree.heading("active", text="启用")
            tree.column("prompt", width=520)
            tree.column("usage", width=80, anchor=tk.CENTER)
            tree.column("active", width=50, anchor=tk.CENTER)

            scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            if self._db_available and table_cls is not None:
                try:
                    for p in table_cls.select():
                        preview = p.prompt_text[:70] + "..." if len(p.prompt_text) > 70 else p.prompt_text
                        tree.insert("", tk.END, values=(
                            preview,
                            p.usage_count or 0,
                            "✓" if p.is_active else "✗"
                        ))
                except Exception as e:
                    messagebox.showerror("错误", f"加载 {tab_name} 提示词数据失败: {e}", parent=win)
            else:
                for text, _ in fallback_list:
                    preview = text[:70] + "..." if len(text) > 70 else text
                    tree.insert("", tk.END, values=(preview, 0, "✓"))

        ttk.Button(win, text="关闭", command=win.destroy).pack(pady=10)

    # ------------------------------------------------------------------
    # 设置保存/加载
    # ------------------------------------------------------------------
    def _save_settings(self):
        """将当前 GUI 参数保存到 AppSettings 表。"""
        if not self._db_available or AppSettings is None:
            messagebox.showinfo("提示", "数据库未连接，设置仅保存在内存中")
            return

        try:
            settings = {
                "style": self.entry_style.get(),
                "subject": self.entry_subject.get(),
                "quality": self.entry_quality.get(),
                "model": self.combo_model.get(),
                "render_size": self.combo_render.get(),
                "depth_backend": self.combo_depth.get(),
                "depth_model": self.combo_depth_model.get(),
                "strength": f"{self.slider_strength.get():.2f}",
                "blend": f"{self.slider_blend.get():.2f}",
                "ema": f"{self.slider_ema.get():.2f}",
                "seed": self.entry_seed.get(),
                "ndi_output": self.entry_ndi.get(),
            }
            # 保存每个 category 的 LoRA 选择
            for category in ["style", "subject", "quality"]:
                sel = self._lora_selection.get(category, {})
                item = sel.get("item")
                settings[f"lora_{category}"] = json.dumps({
                    "name": item["name"] if item else "",
                    "weight": sel.get("weight", 0.0),
                }, ensure_ascii=False)
            for key, value in settings.items():
                AppSettings.set_value(key, value)
            self._log("[SETTINGS] 设置已保存到数据库")
            messagebox.showinfo("成功", "设置已保存到数据库")
        except Exception as e:
            self._log(f"[ERROR] 保存设置失败: {e}")
            messagebox.showerror("错误", f"保存设置失败: {e}")

    def _load_settings(self):
        """从 AppSettings 表加载上次保存的设置。"""
        if not self._db_available or AppSettings is None:
            return

        try:
            # 加载三个 category 输入框
            style = AppSettings.get_value("style")
            if style:
                self.entry_style.delete(0, tk.END)
                self.entry_style.insert(0, style)

            subject = AppSettings.get_value("subject")
            if subject:
                self.entry_subject.delete(0, tk.END)
                self.entry_subject.insert(0, subject)

            quality = AppSettings.get_value("quality")
            if quality:
                self.entry_quality.delete(0, tk.END)
                self.entry_quality.insert(0, quality)

            # 更新合并提示词
            self._update_combined_prompt()

            model = AppSettings.get_value("model")
            if model and model in self.combo_model["values"]:
                self.combo_model.set(model)

            render = AppSettings.get_value("render_size")
            if render:
                # Backward-compat: old integer values like "512" -> "512x512"
                render = str(render).strip()
                if "x" not in render and render in ("320", "384", "512"):
                    render = f"{render}x{render}"
                if render in self.combo_render["values"]:
                    self.combo_render.set(render)

            depth_backend = AppSettings.get_value("depth_backend")
            if depth_backend:
                self.combo_depth.set(depth_backend)

            depth_model = AppSettings.get_value("depth_model")
            if depth_model and depth_model in self.combo_depth_model["values"]:
                self.combo_depth_model.set(depth_model)

            strength = AppSettings.get_value("strength")
            if strength:
                try:
                    self.slider_strength.set(float(strength))
                except ValueError:
                    pass

            blend = AppSettings.get_value("blend")
            if blend:
                try:
                    self.slider_blend.set(float(blend))
                except ValueError:
                    pass

            ema = AppSettings.get_value("ema")
            if ema:
                try:
                    self.slider_ema.set(float(ema))
                except ValueError:
                    pass

            seed = AppSettings.get_value("seed")
            if seed:
                try:
                    self.entry_seed.delete(0, tk.END)
                    self.entry_seed.insert(0, seed)
                except ValueError:
                    pass

            ndi = AppSettings.get_value("ndi_output")
            if ndi:
                self.entry_ndi.delete(0, tk.END)
                self.entry_ndi.insert(0, ndi)

            # 加载 LoRA 选择
            for category in ["style", "subject", "quality"]:
                raw = AppSettings.get_value(f"lora_{category}")
                if not raw:
                    continue
                try:
                    saved = json.loads(raw)
                    name = saved.get("name", "")
                    weight = float(saved.get("weight", 0.0))
                    if name and category in self._lora_widgets:
                        widgets = self._lora_widgets[category]
                        # 找到对应的 combobox 项
                        for idx, (label, item) in enumerate(widgets["choices"]):
                            if item is not None and item["name"] == name:
                                widgets["combo"].set(label)
                                self._on_lora_selection_changed(category)
                                # 在限位内应用保存的权重
                                item = self._lora_selection[category].get("item")
                                if item:
                                    wmin = item["weight_min"]
                                    wmax = item["weight_max"]
                                    weight = max(wmin, min(wmax, weight))
                                    widgets["slider"].set(weight)
                                    self._lora_selection[category]["weight"] = weight
                                    widgets["label"].config(text=f"{weight:.2f}")
                                break
                except Exception as e:
                    self._log(f"[WARN] 加载 {category} LoRA 设置失败: {e}")

            self._log("[SETTINGS] 已从数据库加载设置")
        except Exception as e:
            self._log(f"[WARN] 加载设置失败: {e}")

    # ------------------------------------------------------------------
    # Helpers（原有功能）
    # ------------------------------------------------------------------
    def _log(self, text, tag=""):
        """向日志区域追加一行文本。"""
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, text + "\n", tag)
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

    def _build_cmd(self):
        """根据 GUI 参数构建命令行。"""
        args = []

        # 使用合并后的提示词
        prompt = self._update_combined_prompt()
        args.append(f"--prompt {shlex.quote(prompt)}")

        # 从 combobox 值中提取模型名称
        model_val = self.combo_model.get()
        model_name = model_val.split(" ")[0] if " " in model_val else model_val
        args.append(f"--model {shlex.quote(model_name)}")

        # Parse output resolution: "512x512" -> render-size + output-size, "720x1280" -> render-size 512 + output-size 720x1280
        res_val = self.combo_render.get()
        if res_val == "720x1280":
            args.append("--render-size 512")
            args.append("--output-size 720x1280")
        else:
            size = res_val.split("x")[0] if "x" in res_val else res_val
            args.append(f"--render-size {shlex.quote(size)}")
            args.append(f"--output-size {shlex.quote(size)}")

        args.append(f"--depth-backend {shlex.quote(self.combo_depth.get())}")
        depth_model_val = self.combo_depth_model.get()
        depth_model_name = depth_model_val.split(" ")[0] if " " in depth_model_val else depth_model_val
        args.append(f"--depth-model {shlex.quote(depth_model_name)}")
        args.append(f"--strength {self.slider_strength.get():.2f}")
        args.append(f"--blend {self.slider_blend.get():.2f}")
        args.append(f"--ema {self.slider_ema.get():.2f}")
        args.append(f"--seed {self.entry_seed.get()}")

        # LoRA stack
        lora_stack = self._get_lora_stack()
        for lora in lora_stack:
            args.append(f"--lora {shlex.quote(lora['path'])}")
            args.append(f"--lora-weight {lora['weight']:.2f}")
            args.append(f"--lora-category {shlex.quote(lora['category'])}")

        ndi = self.entry_ndi.get().strip()
        if ndi:
            args.append(f"--ndi-output {shlex.quote(ndi)}")

        extra = self.entry_extra.get().strip()
        if extra:
            args.append(extra)

        full = f"source {VENV_ACTIVATE} && python {SCRIPT_PATH} " + " ".join(args)
        return ["bash", "-c", full]

    # ------------------------------------------------------------------
    # Events（原有功能）
    # ------------------------------------------------------------------
    def _on_start(self):
        """启动 stream_rgbd 子进程。"""
        # 校验：subject 不能为空
        subject = self.entry_subject.get().strip()
        if not subject:
            messagebox.showwarning("提示", "Subject 不能为空，请填写或随机生成一个主题")
            return

        if self._proc is not None and self._proc.poll() is None:
            messagebox.showwarning("提示", "进程已在运行中")
            return

        cmd = self._build_cmd()
        self._log(f"[CMD] {' '.join(cmd)}")
        self._log("-" * 60)

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=PROJECT_DIR,
            )
        except Exception as e:
            self._log(f"[ERROR] 启动失败: {e}")
            return

        self._running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_update_prompt.config(state=tk.NORMAL)
        self.btn_update_seed.config(state=tk.NORMAL)
        self.lbl_current_prompt.config(text=f"当前: {self._update_combined_prompt()[:40]}...", foreground="green")
        self.lbl_current_seed.config(text=f"当前 Seed: {self.entry_seed.get()}", foreground="green")
        self.lbl_status.config(text="状态: 运行中", foreground="green")

        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

    def _read_output(self):
        """在后台线程读取子进程 stdout。"""
        if self._proc is None or self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            if not self._running:
                break
            self.root.after(0, lambda l=line: self._log(l.rstrip()))
        self.root.after(0, self._on_process_exit)

    def _on_process_exit(self):
        """子进程退出后的清理。"""
        if self._proc is not None:
            ret = self._proc.poll()
            self._log("-" * 60)
            self._log(f"[EXIT] 进程退出，返回码: {ret}")
        self._proc = None
        self._running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_update_prompt.config(state=tk.DISABLED)
        self.btn_update_seed.config(state=tk.DISABLED)
        self.lbl_current_prompt.config(text="当前: (未运行)", foreground="gray")
        self.lbl_current_seed.config(text="当前 Seed: 42", foreground="gray")
        self.lbl_status.config(text="状态: 已停止", foreground="gray")

    def _on_update_prompt(self):
        """向运行中的子进程发送新提示词。"""
        if self._proc is None or self._proc.poll() is not None:
            messagebox.showwarning("提示", "进程未运行，无法更新提示词")
            return
        # tk.Text 使用 get("1.0", "end-1c") 获取内容
        new_prompt = self.entry_new_prompt.get("1.0", "end-1c").strip()
        if not new_prompt:
            messagebox.showwarning("提示", "新提示词不能为空")
            return
        try:
            self._proc.stdin.write(new_prompt + "\n")
            self._proc.stdin.flush()
            self.lbl_current_prompt.config(text=f"当前: {new_prompt[:40]}...", foreground="green")
            self._log(f"[PROMPT] 已更新提示词: {new_prompt}")
        except Exception as e:
            self._log(f"[ERROR] 发送提示词失败: {e}")

    def _paste_new_prompt(self):
        """将剪贴板内容粘贴到新提示词框，替换原有内容。"""
        try:
            clipboard = self.root.clipboard_get()
            self.entry_new_prompt.delete("1.0", "end")
            self.entry_new_prompt.insert("1.0", clipboard)
            self._log(f"[PASTE] 已粘贴替换提示词: {clipboard[:50]}...")
        except tk.TclError:
            # 剪贴板为空或非文本内容
            self._log("[WARN] 剪贴板为空，无法粘贴")

    def _on_random_seed(self):
        """生成随机 seed，如果运行中则自动应用。"""
        new_seed = random.randint(0, 2147483647)
        self.entry_seed.delete(0, tk.END)
        self.entry_seed.insert(0, str(new_seed))
        if self._running and self._proc is not None and self._proc.poll() is None:
            self._on_update_seed()

    def _on_update_seed(self):
        """向运行中的子进程发送新 seed。"""
        if self._proc is None or self._proc.poll() is not None:
            messagebox.showwarning("提示", "进程未运行，无法更新 Seed")
            return
        seed_str = self.entry_seed.get().strip()
        try:
            int(seed_str)
        except ValueError:
            messagebox.showwarning("提示", "Seed 必须是整数")
            return
        try:
            self._proc.stdin.write(f"seed:{seed_str}\n")
            self._proc.stdin.flush()
            self.lbl_current_seed.config(text=f"当前 Seed: {seed_str}", foreground="green")
            self._log(f"[SEED] 已更新 Seed: {seed_str}")
        except Exception as e:
            self._log(f"[ERROR] 发送 Seed 失败: {e}")

    def _on_stop(self):
        """停止子进程。"""
        if self._proc is not None and self._proc.poll() is None:
            self._log("[STOP] 发送终止信号...")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._log("[STOP] 强制杀死进程...")
                self._proc.kill()
                self._proc.wait()
        self._running = False
        self._on_process_exit()

    def _on_clear_log(self):
        """清空日志区域。"""
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.delete("1.0", tk.END)
        self.txt_log.config(state=tk.DISABLED)

    def _on_close(self):
        """关闭窗口时的清理。"""
        self._on_stop()
        self.root.destroy()

    def run(self):
        """启动主循环。"""
        self.root.mainloop()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    app = StreamRGBDGUIDB()
    app.run()


if __name__ == "__main__":
    main()
