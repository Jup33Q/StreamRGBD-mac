#!/usr/bin/env python3
"""
Stream NDI GUI — tkinter 包裹的 NDI 输入控制面板
为 camera_ndi 提供可视化参数配置 + 启动/停止 + 日志输出。

Usage:
    python stream_ndi_gui.py
"""
import os
import sys
import subprocess
import threading
import shlex
import random
import sqlite3
import time

# 尝试导入数据库模型（兼容模式：如果 models.py 存在则使用，否则 fallback）
try:
    from models import db, StylePrompt, SubjectPrompt, QualityPrompt
    _DB_OK = True
except Exception:
    _DB_OK = False
    StylePrompt = SubjectPrompt = QualityPrompt = None


# 按 category 拆分的 fallback 提示词（与 camera DB 版保持一致）
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

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
except ImportError:
    print("ERROR: tkinter 不可用。请确认 Python 安装了 Tk 支持。")
    sys.exit(1)

try:
    import NDIlib
    _NDI_OK = True
except Exception:
    _NDI_OK = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from tk_style import GlassWindow, apply_glass_effect
except ImportError:
    GlassWindow = tk.Tk
    apply_glass_effect = lambda *a, **k: None


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(PROJECT_DIR, "python", "camera_ndi.py")
VENV_ACTIVATE = os.path.join(PROJECT_DIR, ".venv", "bin", "activate")


class StreamNDIGUI:
    def __init__(self):
        self._proc = None
        self._reader_thread = None
        self._running = False

        self.root = GlassWindow(material='popover', dark=False, overlay=False)
        self.root.title("Stream NDI 控制台")
        self.root.geometry("900x700")
        self.root.minsize(700, 500)

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

        # Canvas + Scrollbar for left panel
        canvas = tk.Canvas(left_container, width=340, highlightthickness=0)
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

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # --- NDI 输入源（置顶）---
        ndi_scan_frame = ttk.Frame(left)
        ndi_scan_frame.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(ndi_scan_frame, text="NDI 输入源:").pack(side=tk.LEFT)
        self.btn_scan_ndi = tk.Button(
            ndi_scan_frame, text="🔄 扫描", command=self._on_scan_ndi,
            bg="#2196F3", fg="white", activebackground="#1976D2", activeforeground="white",
            font=("Helvetica", 9, "bold"), cursor="hand2",
        )
        self.btn_scan_ndi.pack(side=tk.RIGHT)

        self.combo_ndi_source = ttk.Combobox(left, values=[], state="normal", width=45)
        self.combo_ndi_source.set("")
        self.combo_ndi_source.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(left, text="NDI 输出名称:").pack(anchor=tk.W, pady=(0, 2))
        self.entry_ndi_output = ttk.Entry(left, width=45)
        self.entry_ndi_output.insert(0, "StreamDiffusion-NDI")
        self.entry_ndi_output.pack(fill=tk.X, pady=(0, 8))

        # --- Prompt 按 Category 分组 ---
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

        # --- 合并提示词显示（可复制）---
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

        # --- Runtime Prompt Update（多行 + 粘贴替换）---
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(left, text="新提示词 (运行中可更新):").pack(anchor=tk.W, pady=(0, 2))
        self.entry_new_prompt = tk.Text(
            left, width=40, height=3, wrap=tk.WORD,
            font=("Menlo", 10), bg="#fafafa", fg="#333333",
            insertbackground="#333333", padx=4, pady=4,
        )
        self.entry_new_prompt.insert("1.0", "watercolor painting, soft brushstrokes")
        self.entry_new_prompt.pack(fill=tk.X, pady=(0, 4))

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

        # --- Seed ---
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

        # --- Model ---
        ttk.Label(left, text="Model:").pack(anchor=tk.W, pady=(0, 2))
        self.combo_model = ttk.Combobox(left, values=["sdxs", "sdxs-768"], state="readonly", width=20)
        self.combo_model.set("sdxs")
        self.combo_model.pack(anchor=tk.W, pady=(0, 8))

        # --- Output Resolution ---
        ttk.Label(left, text="Output Resolution:").pack(anchor=tk.W, pady=(0, 2))
        self.combo_render = ttk.Combobox(
            left,
            values=["512x512", "768x768", "384x384", "320x320"],
            state="readonly",
            width=20,
        )
        self.combo_render.set("512x512")
        self.combo_render.pack(anchor=tk.W, pady=(0, 8))

        # --- Depth ---
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(left, text="Depth Backend:").pack(anchor=tk.W, pady=(0, 2))
        self.combo_depth = ttk.Combobox(left, values=["auto", "coreml", "pytorch"], state="readonly", width=20)
        self.combo_depth.set("auto")
        self.combo_depth.pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(left, text="Depth Model:").pack(anchor=tk.W, pady=(0, 2))
        self.combo_depth_model = ttk.Combobox(
            left,
            values=["auto", "da3-small", "da3-base", "da3-large", "da2-small", "da2-base", "da2-large"],
            state="readonly",
            width=20,
        )
        self.combo_depth_model.set("auto")
        self.combo_depth_model.pack(anchor=tk.W, pady=(0, 8))

        # --- Strength / Blend ---
        ttk.Label(left, text="Strength (0.0–1.0):").pack(anchor=tk.W, pady=(0, 2))
        self.entry_strength = ttk.Entry(left, width=20)
        self.entry_strength.insert(0, "0.5")
        self.entry_strength.pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(left, text="Blend (0.0–1.0):").pack(anchor=tk.W, pady=(0, 2))
        self.entry_blend = ttk.Entry(left, width=20)
        self.entry_blend.insert(0, "0.0")
        self.entry_blend.pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(left, text="EMA (0.0–0.99):").pack(anchor=tk.W, pady=(0, 2))
        self.entry_ema = ttk.Entry(left, width=20)
        self.entry_ema.insert(0, "0.4")
        self.entry_ema.pack(anchor=tk.W, pady=(0, 8))

        # --- Extra Args ---
        ttk.Label(left, text="额外参数:").pack(anchor=tk.W, pady=(0, 2))
        self.entry_extra = ttk.Entry(left, width=45)
        self.entry_extra.pack(fill=tk.X, pady=(0, 8))

        # ========== 右侧：日志输出 ==========
        right = ttk.Frame(self.root, padding="10")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(right, text="日志输出:").pack(anchor=tk.W, pady=(0, 5))
        self.txt_log = scrolledtext.ScrolledText(right, wrap=tk.WORD, state=tk.DISABLED, font=("Menlo", 11))
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log(self, text, tag=""):
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
        model_val = self.combo_model.get()
        model_name = model_val.split(" ")[0] if " " in model_val else model_val
        args.append(f"--model {shlex.quote(model_name)}")
        # Parse output resolution: square render + same output
        res_val = self.combo_render.get()
        size = res_val.split("x")[0] if "x" in res_val else res_val
        args.append(f"--render-size {shlex.quote(size)}")
        args.append(f"--output-size {shlex.quote(size)}")

        args.append(f"--strength {self.entry_strength.get()}")
        args.append(f"--blend {self.entry_blend.get()}")
        args.append(f"--ema {self.entry_ema.get()}")
        args.append(f"--seed {self.entry_seed.get()}")

        depth_backend_val = self.combo_depth.get()
        args.append(f"--depth-backend {shlex.quote(depth_backend_val)}")
        depth_model_val = self.combo_depth_model.get()
        depth_model_name = depth_model_val.split(" ")[0] if " " in depth_model_val else depth_model_val
        args.append(f"--depth-model {shlex.quote(depth_model_name)}")

        ndi_source = self.combo_ndi_source.get().strip()
        if ndi_source:
            args.append(f"--ndi-source {shlex.quote(ndi_source)}")

        ndi_output = self.entry_ndi_output.get().strip()
        if ndi_output:
            args.append(f"--ndi-output {shlex.quote(ndi_output)}")

        extra = self.entry_extra.get().strip()
        if extra:
            args.append(extra)

        full = f"source {VENV_ACTIVATE} && exec python {SCRIPT_PATH} " + " ".join(args)
        return ["bash", "-c", full]

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def _on_start(self):
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
        if self._proc is None or self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            if not self._running:
                break
            self.root.after(0, lambda l=line: self._log(l.rstrip()))
        # 进程结束
        self.root.after(0, self._on_process_exit)

    def _on_process_exit(self):
        if self._proc is not None:
            ret = self._proc.poll()
            self._log(f"-" * 60)
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
        """将合并提示词复制到系统剪贴板。"""
        combined = self._update_combined_prompt()
        self.root.clipboard_clear()
        self.root.clipboard_append(combined)
        self.root.update()  # 确保剪贴板更新
        self._log(f"[COPY] 已复制到剪贴板: {combined[:50]}...")
        # 短暂闪烁按钮颜色作为反馈
        self.btn_copy_combined.config(bg="#2E7D32")
        self.root.after(300, lambda: self.btn_copy_combined.config(bg="#4CAF50"))

    def _on_random_category(self, category: str):
        """从指定 category 随机选取一条提示词填入对应输入框。"""
        prompt_text = None

        if _DB_OK:
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
        self._log(f"[PROMPT] {category}: {prompt_text[:50]}...")

    def _on_random_all(self):
        """全部随机：同时随机 style、subject、quality，并更新合并提示词。"""
        for category in ["style", "subject", "quality"]:
            self._on_random_category(category)

    def _on_scan_ndi(self):
        """扫描本地可用的 NDI 源并填充下拉框。"""
        self._log("[NDI] 扫描输入源...")
        self.btn_scan_ndi.config(state=tk.DISABLED)

        def _do_scan():
            sources = []
            if _NDI_OK:
                try:
                    if not NDIlib.initialize():
                        self.root.after(0, lambda: self._log("[NDI] 初始化失败，请确认已安装 NDI SDK"))
                        return
                    finder = NDIlib.find_create_v2()
                    if finder:
                        waited = 0
                        while waited < 3000:
                            NDIlib.find_wait_for_sources(finder, 200)
                            waited += 200
                            found = NDIlib.find_get_current_sources(finder)
                            if found:
                                sources = [s.ndi_name for s in found]
                                break
                        if not sources:
                            found = NDIlib.find_get_current_sources(finder)
                            sources = [s.ndi_name for s in found]
                        NDIlib.find_destroy(finder)
                    NDIlib.destroy()
                except Exception as e:
                    self.root.after(0, lambda: self._log(f"[NDI] 扫描失败: {e}"))
                    return
            else:
                self.root.after(0, lambda: self._log("[NDI] ndi-python 未安装，无法扫描"))
                return

            self.root.after(0, lambda: self._update_ndi_sources(sources))

        threading.Thread(target=_do_scan, daemon=True).start()

    def _update_ndi_sources(self, sources):
        """Update the NDI source combobox with scan results."""
        self.btn_scan_ndi.config(state=tk.NORMAL)
        if not sources:
            self._log("[NDI] 未找到输入源")
            self.combo_ndi_source["values"] = []
            self.combo_ndi_source.set("")
            return

        self.combo_ndi_source["values"] = sources
        self.combo_ndi_source.set(sources[0])
        self._log(f"[NDI] 找到 {len(sources)} 个源:")
        for s in sources:
            self._log(f"  - {s}")

    def _on_random_seed(self):
        new_seed = random.randint(0, 2147483647)
        self.entry_seed.delete(0, tk.END)
        self.entry_seed.insert(0, str(new_seed))
        if self._running and self._proc is not None and self._proc.poll() is None:
            self._on_update_seed()

    def _on_update_seed(self):
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
        if self._proc is not None and self._proc.poll() is None:
            self._log("[STOP] 发送终止信号...")
            self._proc.terminate()
            try:
                # Give the child process time to clean up NDI/CV2 windows.
                self._proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                self._log("[STOP] 进程未在 12s 内退出，强制杀死...")
                self._proc.kill()
                self._proc.wait()
        self._running = False
        self._on_process_exit()

    def _on_clear_log(self):
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.delete("1.0", tk.END)
        self.txt_log.config(state=tk.DISABLED)

    def _on_close(self):
        self._on_stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    app = StreamNDIGUI()
    app.run()


if __name__ == "__main__":
    main()
