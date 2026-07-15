#!/usr/bin/env python3
"""
NDI Source Scanner GUI
扫描局域网内所有可用的 NDI 源（输出），以列表形式展示。

Usage:
    python ndi_scanner_gui.py
    python ndi_scanner_gui.py --auto-refresh 3
"""
import os
import sys
import time
import argparse
import threading

try:
    import NDIlib
except ImportError:
    print("ERROR: NDIlib 未安装。请先安装 ndi-python：")
    print("  uv pip install ndi-python")
    sys.exit(1)

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
except ImportError:
    print("ERROR: tkinter 不可用。请确认 Python 安装了 Tk 支持。")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from tk_style import GlassWindow, apply_glass_effect
except ImportError:
    GlassWindow = tk.Tk
    apply_glass_effect = lambda *a, **k: None


class NDIScannerGUI:
    def __init__(self, auto_refresh_interval=0):
        self.auto_refresh_interval = auto_refresh_interval  # 秒，0=关闭
        self._finder = None
        self._ndi_initialized = False
        self._scanning = False
        self._sources = []
        self._last_scan_time = 0

        # --- GUI 构建 ---
        self.root = GlassWindow(material='popover', dark=False, overlay=False)
        self.root.title("NDI Source Scanner")
        self.root.geometry("700x500")
        self.root.minsize(500, 300)

        # 顶部工具栏
        toolbar = ttk.Frame(self.root, padding="10 5")
        toolbar.pack(fill=tk.X)

        self.btn_scan = ttk.Button(toolbar, text="🔍 扫描", command=self._on_scan)
        self.btn_scan.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_clear = ttk.Button(toolbar, text="清空", command=self._on_clear)
        self.btn_clear.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_status = ttk.Label(toolbar, text="就绪")
        self.lbl_status.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_count = ttk.Label(toolbar, text="源: 0")
        self.lbl_count.pack(side=tk.RIGHT)

        # 列表区域
        list_frame = ttk.Frame(self.root, padding="10 5")
        list_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("name", "address", "url")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("name", text="NDI 源名称")
        self.tree.heading("address", text="地址")
        self.tree.heading("url", text="URL")
        self.tree.column("name", width=250)
        self.tree.column("address", width=200)
        self.tree.column("url", width=200)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # 详情区域
        detail_frame = ttk.LabelFrame(self.root, text="源详情", padding="10 5")
        detail_frame.pack(fill=tk.X, padx=10, pady=5)

        self.txt_detail = scrolledtext.ScrolledText(detail_frame, height=6, wrap=tk.WORD)
        self.txt_detail.pack(fill=tk.BOTH, expand=True)

        # 底部状态栏
        bottom = ttk.Frame(self.root, padding="5 5")
        bottom.pack(fill=tk.X, side=tk.BOTTOM)

        self.lbl_bottom = ttk.Label(bottom, text="点击 [扫描] 开始搜索 NDI 源")
        self.lbl_bottom.pack(side=tk.LEFT)

        # 绑定事件
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 初始化 NDI
        if not NDIlib.initialize():
            self.lbl_status.config(text="NDI 初始化失败", foreground="red")
            self.btn_scan.config(state=tk.DISABLED)
        else:
            self._ndi_initialized = True
            self.lbl_status.config(text="NDI 已初始化", foreground="green")

        # 自动刷新
        if self.auto_refresh_interval > 0:
            self._schedule_auto_refresh()

    def _schedule_auto_refresh(self):
        self._on_scan()
        self.root.after(int(self.auto_refresh_interval * 1000), self._schedule_auto_refresh)

    def _on_scan(self):
        if self._scanning:
            return
        self._scanning = True
        self.btn_scan.config(state=tk.DISABLED)
        self.lbl_status.config(text="扫描中...")
        self.lbl_bottom.config(text="正在搜索 NDI 源...")

        # 在后台线程扫描，避免阻塞 GUI
        t = threading.Thread(target=self._do_scan, daemon=True)
        t.start()

    def _do_scan(self):
        finder = NDIlib.find_create_v2()
        if not finder:
            self.root.after(0, lambda: self._scan_done([], "创建 NDI finder 失败"))
            return

        sources = []
        waited = 0
        step = 500       # ms
        timeout = 10000  # 总超时 10 秒
        found_names = set()

        while waited < timeout:
            NDIlib.find_wait_for_sources(finder, step)
            waited += step
            current = NDIlib.find_get_current_sources(finder)
            if current:
                for s in current:
                    if s.ndi_name not in found_names:
                        found_names.add(s.ndi_name)
                        sources.append(s)
                # 继续扫描，不 break，收集所有源直到超时

        NDIlib.find_destroy(finder)
        self.root.after(0, lambda: self._scan_done(sources, None, waited))

    def _scan_done(self, sources, error, scan_time_ms=0):
        self._scanning = False
        self._sources = sources
        self._last_scan_time = time.time()

        self.tree.delete(*self.tree.get_children())
        self.txt_detail.delete("1.0", tk.END)

        if error:
            self.lbl_status.config(text=f"错误: {error}", foreground="red")
            self.lbl_count.config(text="源: 0")
            self.lbl_bottom.config(text=error)
        elif not sources:
            self.lbl_status.config(text="未找到 NDI 源", foreground="orange")
            self.lbl_count.config(text="源: 0")
            self.lbl_bottom.config(text=f"扫描完成，未找到 NDI 源 (耗时 {scan_time_ms//1000}s)")
        else:
            self.lbl_status.config(text=f"找到 {len(sources)} 个源", foreground="green")
            self.lbl_count.config(text=f"源: {len(sources)}")
            self.lbl_bottom.config(text=f"最后扫描: {time.strftime('%H:%M:%S')}  (耗时 {scan_time_ms//1000}s)")

            for i, s in enumerate(sources):
                self.tree.insert("", tk.END, iid=str(i), values=(
                    s.ndi_name,
                    s.url_address or "",
                    "",
                ))

        self.btn_scan.config(state=tk.NORMAL)

    def _on_select(self, event):
        selection = self.tree.selection()
        if not selection:
            return
        idx = int(selection[0])
        if idx < 0 or idx >= len(self._sources):
            return

        s = self._sources[idx]
        self.txt_detail.delete("1.0", tk.END)
        self.txt_detail.insert(tk.END, f"名称: {s.ndi_name}\n")
        self.txt_detail.insert(tk.END, f"URL/地址: {s.url_address or 'N/A'}\n")

    def _on_clear(self):
        self.tree.delete(*self.tree.get_children())
        self.txt_detail.delete("1.0", tk.END)
        self._sources = []
        self.lbl_count.config(text="源: 0")
        self.lbl_status.config(text="已清空")

    def _on_close(self):
        if self._ndi_initialized:
            try:
                NDIlib.destroy()
            except Exception:
                pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="NDI Source Scanner GUI")
    parser.add_argument("--auto-refresh", type=int, default=0, metavar="SEC",
                        help="自动刷新间隔（秒），0=关闭")
    args = parser.parse_args()

    app = NDIScannerGUI(auto_refresh_interval=args.auto_refresh)
    app.run()


if __name__ == "__main__":
    main()
