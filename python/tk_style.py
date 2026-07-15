"""
tk_style.py — 跨平台 tkinter 窗口样式模块

支持 macOS、Windows 以及 Linux 的窗口配色与样式控制。

Usage:
    from tk_style import GlassWindow, apply_glass_effect

    # 方式一：继承 GlassWindow
    root = GlassWindow(material='popover', dark=True, overlay=False)

    # 方式二：对已有窗口应用效果
    root = tk.Tk()
    apply_glass_effect(root, material='popover', dark=True, overlay=False)
"""
import platform
import threading
import time

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    raise ImportError("tkinter 不可用。请确认 Python 安装了 Tk 支持。")

_SYSTEM = platform.system()

# ---------------------------------------------------------------------------
# macOS: NSVisualEffectView
# ---------------------------------------------------------------------------
if _SYSTEM == 'Darwin':
    try:
        from AppKit import (
            NSApplication,
            NSVisualEffectView,
            NSMakeRect,
            NSVisualEffectMaterialPopover,
            NSVisualEffectMaterialMenu,
            NSVisualEffectMaterialSidebar,
            NSVisualEffectMaterialHUDWindow,
            NSVisualEffectMaterialFullScreenUI,
            NSVisualEffectStateActive,
            NSWindowBelow,
            NSViewWidthSizable,
            NSViewHeightSizable,
            NSFullSizeContentViewWindowMask,
            NSVisualEffectBlendingModeBehindWindow,
            NSFloatingWindowLevel,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorStationary,
            NSWindowCollectionBehaviorIgnoresCycle,
        )
        _MACOS_OK = True
    except ImportError:
        _MACOS_OK = False

    def _macos_material(name: str):
        mapping = {
            'popover': NSVisualEffectMaterialPopover,
            'menu': NSVisualEffectMaterialMenu,
            'sidebar': NSVisualEffectMaterialSidebar,
            'hud': NSVisualEffectMaterialHUDWindow,
            'fullscreen': NSVisualEffectMaterialFullScreenUI,
        }
        return mapping.get(name.lower(), NSVisualEffectMaterialPopover)

    def _macos_apply_glass(widget, material='popover', dark=True, overlay=False):
        if not _MACOS_OK:
            return False

        root = widget.winfo_toplevel()
        title = root.title()

        # 必须等窗口真正创建后才能找到 NSWindow
        def _do_apply():
            # 尝试在事件循环中执行，确保窗口已映射
            root.after(100, lambda: _macos_do_apply_inner(root, title, material, dark, overlay))

        root.after_idle(_do_apply)
        return True

    def _macos_do_apply_inner(root, title, material, dark, overlay):
        """macOS 上不使用毛玻璃，改为传统黑白灰配色。"""
        # 仅设置窗口背景色，不添加任何 NSVisualEffectView 或透明效果
        if dark:
            root.configure(bg='#1e1e1e')
            try:
                root.option_add('*Foreground', '#e0e0e0')
            except Exception:
                pass
        else:
            root.configure(bg='#f5f5f5')
        print(f"[tk_style] macOS traditional style: dark={dark}")


# ---------------------------------------------------------------------------
# Windows: BlurWindow
# ---------------------------------------------------------------------------
elif _SYSTEM == 'Windows':
    try:
        from BlurWindow.blurWindow import GlobalBlur
        _WINDOWS_OK = True
    except ImportError:
        _WINDOWS_OK = False

    def _windows_apply_glass(widget, material='popover', dark=True, overlay=False):
        if not _WINDOWS_OK:
            return False
        root = widget.winfo_toplevel()
        hwnd = root.winfo_id()
        try:
            GlobalBlur(hwnd, hexColor=False, Acrylic=True, Dark=dark, QWidget=None)
            if overlay:
                root.attributes('-topmost', True)
            print(f"[tk_style] Windows glass applied: dark={dark}, overlay={overlay}")
            return True
        except Exception as e:
            print(f"[tk_style] Windows glass failed: {e}")
            return False


# ---------------------------------------------------------------------------
# Linux / Fallback
# ---------------------------------------------------------------------------
else:
    def _linux_apply_glass(widget, material='popover', dark=True, overlay=False):
        # Linux 下仅做简单的半透明效果
        root = widget.winfo_toplevel()
        root.attributes('-alpha', 0.92)
        if overlay:
            root.attributes('-topmost', True)
        print(f"[tk_style] Linux fallback (alpha) applied: overlay={overlay}")
        return True


# ---------------------------------------------------------------------------
# 统一接口
# ---------------------------------------------------------------------------
_MATERIALS = {'popover', 'menu', 'sidebar', 'hud', 'fullscreen'}


def apply_glass_effect(widget, material='popover', dark=True, overlay=False):
    """
    为 tkinter 窗口/控件应用毛玻璃效果。

    Args:
        widget: 任意 tkinter 控件，会取其顶层窗口 (winfo_toplevel)。
        material: macOS 材质类型 ('popover', 'menu', 'sidebar', 'hud', 'fullscreen')。
        dark: 是否使用暗色主题（影响文字/图标颜色）。
        overlay: 是否设为屏幕叠加层（始终置顶、忽略 Space/桌面切换）。

    Returns:
        bool: 是否成功应用效果。
    """
    if material.lower() not in _MATERIALS:
        raise ValueError(f"Unknown material: {material}. Choose from {_MATERIALS}")

    if _SYSTEM == 'Darwin':
        return _macos_apply_glass(widget, material=material, dark=dark, overlay=overlay)
    elif _SYSTEM == 'Windows':
        return _windows_apply_glass(widget, material=material, dark=dark, overlay=overlay)
    else:
        return _linux_apply_glass(widget, material=material, dark=dark, overlay=overlay)


class GlassWindow(tk.Tk):
    """
    带有原生毛玻璃效果的 tkinter 窗口。

    在 macOS 上使用 NSVisualEffectView，在 Windows 上使用 BlurWindow，
    在其他平台上使用半透明回退。

    Args:
        material: macOS 材质 (popover/menu/sidebar/hud/fullscreen)。
        dark: 暗色主题。
        overlay: 屏幕叠加层模式（置顶、忽略 Space）。
        **kwargs: 透传给 tk.Tk。
    """

    def __init__(self, material='popover', dark=True, overlay=False, **kwargs):
        super().__init__(**kwargs)
        self._glass_material = material
        self._glass_dark = dark
        self._glass_overlay = overlay
        self._glass_applied = False

        # 延迟应用，确保窗口已创建
        self.after(50, self._apply_glass)

    def _apply_glass(self):
        if not self._glass_applied:
            self._glass_applied = apply_glass_effect(
                self,
                material=self._glass_material,
                dark=self._glass_dark,
                overlay=self._glass_overlay,
            )

    def set_glass(self, material=None, dark=None, overlay=None):
        """运行时重新配置毛玻璃参数（需要重新创建窗口才能生效）。"""
        if material is not None:
            self._glass_material = material
        if dark is not None:
            self._glass_dark = dark
        if overlay is not None:
            self._glass_overlay = overlay


# ---------------------------------------------------------------------------
# Screen Overlay 辅助
# ---------------------------------------------------------------------------
class ScreenOverlay(tk.Toplevel):
    """
    屏幕叠加层：无边框、半透明、置顶的浮动窗口。
    可用作 HUD、通知面板、快捷工具栏等。
    """

    def __init__(self, master=None, width=400, height=300, x=None, y=None,
                 alpha=0.85, material='hud', dark=True, **kwargs):
        super().__init__(master, **kwargs)

        self.overrideredirect(True)  # 无边框
        self.attributes('-alpha', alpha)
        self.attributes('-topmost', True)

        # 默认居中
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = x if x is not None else (sw - width) // 2
        y = y if y is not None else (sh - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

        # 应用毛玻璃
        apply_glass_effect(self, material=material, dark=dark, overlay=True)

        # 允许拖动
        self._drag_data = {'x': 0, 'y': 0}
        self.bind('<Button-1>', self._on_drag_start)
        self.bind('<B1-Motion>', self._on_drag)

        # ESC 关闭
        self.bind('<Escape>', lambda e: self.destroy())

    def _on_drag_start(self, event):
        self._drag_data['x'] = event.x
        self._drag_data['y'] = event.y

    def _on_drag(self, event):
        dx = event.x - self._drag_data['x']
        dy = event.y - self._drag_data['y']
        x = self.winfo_x() + dx
        y = self.winfo_y() + dy
        self.geometry(f"+{x}+{y}")
