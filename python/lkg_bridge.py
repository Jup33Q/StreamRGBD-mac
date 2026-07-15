#!/usr/bin/env python3
"""
LKGRGBDRenderer — Looking Glass 显示器 RGBD 实时渲染封装

基于 Looking-Glass Bridge Python SDK，在独立线程中运行 GLFW + OpenGL
渲染循环，通过线程安全队列接收 numpy RGBA 帧并实时显示到 LKG 显示器。

依赖:
    pip install bridge-python-sdk  # 已包含 glfw, PyOpenGL

Usage:
    renderer = LKGRGBDRenderer(
        app_name="StreamDiffusion-LKG",
        depth_loc=2,      # depth 在 alpha 通道 (RGBA)
        depthiness=1.0,
        focus=0.0,
    )
    renderer.start()      # 初始化 + 启动渲染线程

    # 主循环中每帧调用:
    renderer.update_frame(rgbd_frame)   # HxWx4 uint8 RGBA

    renderer.toggle_visibility()        # 显示 / 隐藏窗口
    renderer.close()                    # 安全关闭
"""
import os
import sys
import time
import threading
import queue
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports — gracefully degrade if SDK not installed
# ---------------------------------------------------------------------------
try:
    import glfw
    from OpenGL import GL
    from bridge_python_sdk import BridgeAPI, PixelFormats
    _LKG_AVAILABLE = True
except Exception as e:
    _LKG_AVAILABLE = False
    _LKG_IMPORT_ERROR = str(e)


class LKGRGBDRenderer:
    """
    将 RGBD 帧实时渲染到 Looking Glass 显示器。

    Attributes:
        width, height: 当前输入帧尺寸（首次 update_frame 后确定）
        running:       渲染线程是否正在运行
        visible:       窗口当前是否可见
    """

    def __init__(
        self,
        app_name: str = "LKGRGBDRenderer",
        depth_loc: int = 2,
        depthiness: float = 1.0,
        focus: float = 0.0,
        max_queue: int = 2,
        debug: bool = False,
    ):
        """
        Args:
            app_name:    Bridge 初始化名称
            depth_loc:   深度通道位置
                         0=top, 1=bottom, 2=left, 3=right.
                         对于 RGBA（depth 在 alpha），实际使用 quilt 内的
                         depth location 映射；默认 2 对应 left（alpha 通道
                         被 bridge 内部处理为 depth）。
            depthiness:  深度强度乘数 (0–3)
            focus:       焦点偏移 (-1–1)
            max_queue:   帧队列最大长度，超限时丢弃旧帧
            debug:       是否打印 Bridge 调试信息
        """
        if not _LKG_AVAILABLE:
            raise ImportError(
                f"Looking-Glass Bridge SDK 不可用。{_LKG_IMPORT_ERROR}\n"
                "请安装:  uv pip install 'bridge-python-sdk @ "
                "git+https://github.com/Looking-Glass/bridge-python-sdk'"
            )

        self.app_name = app_name
        self.depth_loc = depth_loc
        self.depthiness = depthiness
        self.focus = focus
        self.max_queue = max_queue
        self.debug = debug

        self._frame_queue = queue.Queue(maxsize=max_queue)
        self._thread = None
        self._running = False
        self._closing = False

        self.width = 0
        self.height = 0
        self.visible = True

        # 内部状态（仅在渲染线程中访问）
        self._bridge = None
        self._window = None
        self._texture = None
        self._quilt_w = 0
        self._quilt_h = 0
        self._cols = 0
        self._rows = 0
        self._aspect = 1.0
        self._normalized_focus = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """初始化 Bridge + GLFW，启动渲染线程。返回是否成功。"""
        if self._running:
            return True
        self._closing = False
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        # 等待初始化完成（最多 10 秒）
        for _ in range(100):
            if self._running:
                return True
            time.sleep(0.1)
        return False

    def update_frame(self, frame: np.ndarray) -> bool:
        """
        将一帧 RGBA 图像推入渲染队列。

        Args:
            frame: HxWx4 uint8 numpy array，RGB 在前 3 通道，
                   depth 在第 4 通道（alpha）。

        Returns:
            是否成功放入队列（False = 队列满，旧帧被丢弃后仍放入新帧）
        """
        if not self._running or self._closing:
            return False

        if frame.ndim != 3 or frame.shape[2] != 4:
            raise ValueError(f"Frame must be HxWx4 uint8, got shape {frame.shape}")
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        h, w = frame.shape[:2]
        self.width = w
        self.height = h

        # 队列满时丢弃旧帧，确保始终显示最新内容
        while self._frame_queue.qsize() >= self.max_queue:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        try:
            self._frame_queue.put_nowait(frame)
            return True
        except queue.Full:
            return False

    def toggle_visibility(self) -> None:
        """切换 LKG 窗口的显示 / 隐藏状态。"""
        if self._bridge and self._window:
            self.visible = not self.visible
            self._bridge.show_window(self._window, self.visible)

    def show(self) -> None:
        """显示 LKG 窗口。"""
        if self._bridge and self._window:
            self.visible = True
            self._bridge.show_window(self._window, True)

    def hide(self) -> None:
        """隐藏 LKG 窗口。"""
        if self._bridge and self._window:
            self.visible = False
            self._bridge.show_window(self._window, False)

    def close(self) -> None:
        """安全关闭渲染线程并释放资源。"""
        if not self._running:
            return
        self._closing = True
        # 唤醒队列，让渲染线程注意到关闭信号
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=5.0)
        self._running = False
        print("[LKG] Renderer closed.")

    # ------------------------------------------------------------------
    # 渲染线程（内部）
    # ------------------------------------------------------------------
    def _render_loop(self):
        """渲染线程主循环：初始化 GL → 每帧接收 RGBD → 渲染到 LKG。"""
        try:
            ok = self._init_gl()
            if not ok:
                print("[LKG] GL/Bridge 初始化失败，渲染线程退出。")
                return

            self._running = True
            print(f"[LKG] Renderer started ({self.width}x{self.height}).")

            while not self._closing:
                # ---- 获取最新帧 ----
                try:
                    frame = self._frame_queue.get(timeout=0.1)
                except queue.Empty:
                    # 没有新帧，继续 poll events
                    glfw.poll_events()
                    continue

                if frame is None:
                    break  # 收到关闭信号

                # ---- 更新 texture ----
                self._upload_frame(frame)

                # ---- 渲染到 LKG ----
                self._draw_frame()

                # ---- 处理事件 ----
                glfw.poll_events()
                if glfw.window_should_close(self._glfw_window):
                    break

        except Exception as e:
            print(f"[LKG] Render loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup_gl()
            self._running = False

    # ------------------------------------------------------------------
    # GL / Bridge 初始化与清理
    # ------------------------------------------------------------------
    def _init_gl(self) -> bool:
        """初始化 GLFW、OpenGL context、Bridge、texture。"""
        # --- GLFW ---
        if not glfw.init():
            print("[LKG] GLFW init failed")
            return False

        gl_major, gl_minor = 4, 1
        if sys.platform == "darwin" and (gl_major, gl_minor) > (4, 1):
            gl_major, gl_minor = 4, 1

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, gl_major)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, gl_minor)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        if sys.platform == "darwin":
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        # 初始不可见，等待 toggle
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        self._glfw_window = glfw.create_window(1, 1, self.app_name, None, None)
        if not self._glfw_window:
            print("[LKG] Failed to create GLFW window")
            glfw.terminate()
            return False

        glfw.make_context_current(self._glfw_window)
        glfw.swap_interval(0)  # 禁用 vsync，自由帧率

        # --- Bridge ---
        self._bridge = BridgeAPI(debug=self.debug)
        if not self._bridge.initialize(self.app_name):
            print("[LKG] Bridge initialize failed")
            glfw.destroy_window(self._glfw_window)
            glfw.terminate()
            return False

        self._window = self._bridge.instance_window_gl(-1)
        if self._window == 0:
            print("[LKG] Bridge instance_window_gl returned 0")
            glfw.destroy_window(self._glfw_window)
            glfw.terminate()
            return False

        # --- Quilt settings ---
        asp, qw, qh, cols, rows = self._bridge.get_default_quilt_settings(self._window)
        self._aspect = float(asp)
        self._quilt_w = qw
        self._quilt_h = qh
        self._cols = cols
        self._rows = rows
        if self.debug:
            print(f"[LKG] Quilt: {qw}x{qh}, cols={cols}, rows={rows}, aspect={self._aspect}")

        # --- Focus normalization (同 SDK 示例) ---
        focus_min = 0.005
        focus_max = -0.007
        bridge_focus = self.focus * self.depthiness
        self._normalized_focus = focus_min + ((bridge_focus + 1.0) / 2.0) * (focus_max - focus_min)

        # --- Texture placeholder ---
        self._texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

        # 窗口默认可见（可由 toggle 控制）
        if self.visible:
            self._bridge.show_window(self._window, True)

        return True

    def _upload_frame(self, frame: np.ndarray):
        """将 numpy RGBA 帧上传到 OpenGL texture。"""
        h, w = frame.shape[:2]

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)

        if self.width != w or self.height != h:
            # 尺寸变化，重新分配 texture
            self.width = w
            self.height = h
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8,
                w, h, 0,
                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, frame,
            )
        else:
            # 子区域更新（更快）
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D, 0, 0, 0,
                w, h,
                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, frame,
            )

    def _draw_frame(self):
        """调用 Bridge 将 texture 渲染到 LKG 显示器。"""
        self._bridge.draw_interop_rgbd_texture_gl(
            self._window,
            self._texture,
            PixelFormats.RGBA,
            self.width,
            self.height,
            self._quilt_w,
            self._quilt_h,
            self._cols,
            self._rows,
            self._aspect,
            self._normalized_focus,
            self.depthiness,
            1.0,             # zoom (unused)
            self.depth_loc,
        )

    def _cleanup_gl(self):
        """释放所有 GL / Bridge / GLFW 资源。"""
        if self._texture:
            try:
                GL.glDeleteTextures(1, [self._texture])
            except Exception:
                pass
            self._texture = None

        if self._window:
            try:
                self._bridge.show_window(self._window, False)
            except Exception:
                pass
            self._window = None

        if self._bridge:
            try:
                self._bridge.uninitialize()
            except Exception:
                pass
            self._bridge = None

        if hasattr(self, '_glfw_window') and self._glfw_window:
            try:
                glfw.destroy_window(self._glfw_window)
            except Exception:
                pass
            self._glfw_window = None

        try:
            glfw.terminate()
        except Exception:
            pass


# =======================================================================
# 便捷工厂函数
# =======================================================================
def create_lkg_renderer(**kwargs) -> "LKGRGBDRenderer | None":
    """
    安全地创建 LKGRGBDRenderer，SDK 不可用时返回 None 而非抛出异常。
    """
    if not _LKG_AVAILABLE:
        warnings.warn(
            f"Looking-Glass Bridge SDK 不可用: {_LKG_IMPORT_ERROR}\n"
            "LKG 输出将被禁用。"
        )
        return None
    return LKGRGBDRenderer(**kwargs)


# =======================================================================
# CLI 测试
# =======================================================================
def _test_main():
    """加载一张 RGBD 图片循环播放，用于测试 LKG 连接。"""
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(description="LKG RGBD renderer test")
    parser.add_argument("image", help="Path to RGBA image (depth in alpha)")
    parser.add_argument("--depthiness", type=float, default=1.0)
    parser.add_argument("--focus", type=float, default=0.0)
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGBA")
    frame = np.array(img, dtype=np.uint8)

    renderer = LKGRGBDRenderer(
        depthiness=args.depthiness,
        focus=args.focus,
        debug=True,
    )
    if not renderer.start():
        print("Failed to start LKG renderer")
        sys.exit(1)

    print("Press Ctrl+C to stop")
    try:
        while True:
            renderer.update_frame(frame)
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()


if __name__ == "__main__":
    _test_main()
