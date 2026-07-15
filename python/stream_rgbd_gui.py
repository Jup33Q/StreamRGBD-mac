#!/usr/bin/env python3
"""
Stream RGBD GUI — tkinter 包裹的控制面板
为 stream_rgbd 提供可视化参数配置 + 启动/停止 + 日志输出。

Usage:
    python stream_rgbd_gui.py
"""
import os
import sys
import subprocess
import threading
import shlex

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
except ImportError:
    print("ERROR: tkinter 不可用。请确认 Python 安装了 Tk 支持。")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from tk_style import GlassWindow, apply_glass_effect
except ImportError:
    GlassWindow = tk.Tk
    apply_glass_effect = lambda *a, **k: None


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(PROJECT_DIR, "python", "camera_rgbd.py")
VENV_ACTIVATE = os.path.join(PROJECT_DIR, ".venv", "bin", "activate")


class StreamRGBDGUI:
    def __init__(self):
        self._proc = None
        self._reader_thread = None
        self._running = False

        self.root = GlassWindow(material='popover', dark=False, overlay=False)
        self.root.title("Stream RGBD 控制台")
        self.root.geometry("900x700")
        self.root.minsize(700, 500)

        # ========== 左侧：参数面板 ==========
        left = ttk.Frame(self.root, padding="10")
        left.pack(side=tk.LEFT, fill=tk.Y)

        # --- Prompt ---
        ttk.Label(left, text="Prompt:").pack(anchor=tk.W, pady=(0, 2))
        self.entry_prompt = ttk.Entry(left, width=45)
        self.entry_prompt.insert(0, "oil painting style, masterpiece")
        self.entry_prompt.pack(fill=tk.X, pady=(0, 8))

        # --- Model ---
        ttk.Label(left, text="Model:").pack(anchor=tk.W, pady=(0, 2))
        self.combo_model = ttk.Combobox(left, values=["sdxs", "sd-turbo"], state="readonly", width=20)
        self.combo_model.set("sdxs")
        self.combo_model.pack(anchor=tk.W, pady=(0, 8))

        # --- Render Size ---
        ttk.Label(left, text="Render Size:").pack(anchor=tk.W, pady=(0, 2))
        self.combo_render = ttk.Combobox(left, values=[320, 384, 512], state="readonly", width=20)
        self.combo_render.set(512)
        self.combo_render.pack(anchor=tk.W, pady=(0, 8))

        # --- Depth ---
        ttk.Label(left, text="Depth Backend:").pack(anchor=tk.W, pady=(0, 2))
        self.combo_depth = ttk.Combobox(left, values=["auto", "coreml", "pytorch"], state="readonly", width=20)
        self.combo_depth.set("pytorch")
        self.combo_depth.pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(left, text="Depth Model:").pack(anchor=tk.W, pady=(0, 2))
        self.combo_depth_model = ttk.Combobox(left, values=["auto", "da3-small", "da2-small"], state="readonly", width=20)
        self.combo_depth_model.set("da2-small")
        self.combo_depth_model.pack(anchor=tk.W, pady=(0, 8))

        # --- NDI ---
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(left, text="NDI 输出名称:").pack(anchor=tk.W, pady=(0, 2))
        self.entry_ndi = ttk.Entry(left, width=45)
        self.entry_ndi.insert(0, "StreamDiffusion-RGBD")
        self.entry_ndi.pack(fill=tk.X, pady=(0, 8))

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

        # --- Buttons ---
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        self.btn_start = ttk.Button(btn_frame, text="▶ 启动", command=self._on_start)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_stop = ttk.Button(btn_frame, text="⏹ 停止", command=self._on_stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_clear = ttk.Button(btn_frame, text="清空日志", command=self._on_clear_log)
        self.btn_clear.pack(side=tk.LEFT)

        # --- Status ---
        self.lbl_status = ttk.Label(left, text="状态: 就绪", foreground="gray")
        self.lbl_status.pack(anchor=tk.W, pady=(15, 0))

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
        cmd = ["bash", "-c", f"source {VENV_ACTIVATE} && python {SCRIPT_PATH}"]

        args = []
        args.append(f"--prompt {shlex.quote(self.entry_prompt.get())}")
        args.append(f"--model {self.combo_model.get()}")
        args.append(f"--render-size {self.combo_render.get()}")
        args.append(f"--depth-backend {self.combo_depth.get()}")
        args.append(f"--depth-model {self.combo_depth_model.get()}")
        args.append(f"--strength {self.entry_strength.get()}")
        args.append(f"--blend {self.entry_blend.get()}")
        args.append(f"--ema {self.entry_ema.get()}")

        ndi = self.entry_ndi.get().strip()
        if ndi:
            args.append(f"--ndi-output {shlex.quote(ndi)}")

        extra = self.entry_extra.get().strip()
        if extra:
            args.append(extra)

        # 重新构造完整命令
        full = f"source {VENV_ACTIVATE} && python {SCRIPT_PATH} " + " ".join(args)
        return ["bash", "-c", full]

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def _on_start(self):
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
        self.lbl_status.config(text="状态: 已停止", foreground="gray")

    def _on_stop(self):
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
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.delete("1.0", tk.END)
        self.txt_log.config(state=tk.DISABLED)

    def _on_close(self):
        self._on_stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    app = StreamRGBDGUI()
    app.run()


if __name__ == "__main__":
    main()
