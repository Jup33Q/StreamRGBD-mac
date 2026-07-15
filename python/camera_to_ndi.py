#!/usr/bin/env python3
"""
Camera to NDI — 基础摄像头 → NDI 转发脚本
零 AI 处理，仅做摄像头捕获 + NDI 推流。

Usage:
    python camera_to_ndi.py
    python camera_to_ndi.py --camera 1 --ndi-output "MyCamera" --width 1280 --height 720 --fps 30
    python camera_to_ndi.py --no-preview          # 无预览窗口，纯后台转发
"""
import os
import sys
import time
import argparse
import threading
import numpy as np
import cv2
import NDIlib


def list_cameras():
    """列出系统上可用的摄像头设备。"""
    print("\n--- 扫描可用摄像头 ---")
    available = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [{idx}] {w}x{h}")
            available.append(idx)
        cap.release()
    if not available:
        print("  未找到可用摄像头")
    return available


def create_ndi_sender(name):
    """创建 NDI 发送器。"""
    send_settings = NDIlib.SendCreate(p_ndi_name=name)
    sender = NDIlib.send_create(send_settings)
    if not sender:
        raise RuntimeError(f"NDI 发送器创建失败: '{name}'")
    print(f"  NDI 发送器已创建: '{name}'")
    return sender


def bgr_to_ndi(frame_bgr):
    """OpenCV BGR → NDI BGRX 视频帧。"""
    h, w = frame_bgr.shape[:2]
    bgrx = np.zeros((h, w, 4), dtype=np.uint8)
    bgrx[:, :, :3] = frame_bgr
    bgrx[:, :, 3] = 255

    vf = NDIlib.VideoFrameV2()
    vf.xres = w
    vf.yres = h
    vf.FourCC = NDIlib.FourCCVideoType.FOURCC_VIDEO_TYPE_BGRX
    vf.line_stride_in_bytes = w * 4
    vf.frame_rate_N = 30
    vf.frame_rate_D = 1
    vf.data = bgrx
    return vf


class CameraToNDI:
    def __init__(self, camera_id=0, ndi_name="Camera-NDI",
                 width=1280, height=720, fps=30, show_preview=True):
        self.camera_id = camera_id
        self.ndi_name = ndi_name
        self.width = width
        self.height = height
        self.fps = fps
        self.show_preview = show_preview
        self.running = False
        self._cap = None
        self._sender = None
        self._frame_count = 0
        self._start_time = None

    def run(self):
        # --- 初始化 NDI ---
        if not NDIlib.initialize():
            print("ERROR: NDI 初始化失败。请确认已安装 NDI SDK。")
            return

        # --- 打开摄像头 ---
        print(f"\n正在打开摄像头 {self.camera_id}...")
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            print(f"ERROR: 无法打开摄像头 {self.camera_id}")
            available = list_cameras()
            if available:
                print(f"提示: 尝试使用 --camera {available[0]}")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 预热
        for _ in range(10):
            ret, _ = self._cap.read()
            if ret:
                break
            time.sleep(0.05)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        print(f"  摄像头已就绪: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")

        # --- 创建 NDI 发送器 ---
        self._sender = create_ndi_sender(self.ndi_name)

        # --- 启动主循环 ---
        self.running = True
        self._start_time = time.perf_counter()

        if self.show_preview:
            cv2.namedWindow("Camera → NDI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camera → NDI", actual_w // 2, actual_h // 2)

        print(f"\n{'='*50}")
        print(f"Camera {self.camera_id} → NDI '{self.ndi_name}'")
        print(f"分辨率: {actual_w}x{actual_h} | 预览: {'开' if self.show_preview else '关'}")
        print("按 'q' 退出")
        print(f"{'='*50}\n")

        while self.running:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue

            # 如有需要，缩放至目标分辨率
            if frame.shape[1] != actual_w or frame.shape[0] != actual_h:
                frame = cv2.resize(frame, (actual_w, actual_h))

            # 发送 NDI
            ndi_frame = bgr_to_ndi(frame)
            NDIlib.send_send_video_v2(self._sender, ndi_frame)
            self._frame_count += 1

            # 预览
            if self.show_preview:
                fps = self._frame_count / (time.perf_counter() - self._start_time)
                display = frame.copy()
                cv2.putText(display, f"FPS: {fps:.1f} | NDI: {self.ndi_name}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Camera → NDI", display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

        # --- 清理 ---
        self._cleanup()

    def _cleanup(self):
        print("\n正在关闭...")
        elapsed = time.perf_counter() - self._start_time
        avg_fps = self._frame_count / elapsed if elapsed > 0 else 0
        print(f"  总计发送: {self._frame_count} 帧 ({avg_fps:.1f} FPS)")

        if self._cap:
            self._cap.release()
        if self._sender:
            try:
                NDIlib.send_destroy(self._sender)
            except Exception:
                pass
        try:
            NDIlib.destroy()
        except Exception:
            pass
        if self.show_preview:
            cv2.destroyAllWindows()
        print("  已关闭。")


def main():
    parser = argparse.ArgumentParser(description="Camera → NDI 基础转发")
    parser.add_argument("--camera", type=int, default=0,
                        help="摄像头设备 ID (默认: 0)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="列出可用摄像头并退出")
    parser.add_argument("--ndi-output", type=str, default="Camera-NDI",
                        help="NDI 输出源名称 (默认: Camera-NDI)")
    parser.add_argument("--width", type=int, default=1280,
                        help="目标宽度 (默认: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="目标高度 (默认: 720)")
    parser.add_argument("--fps", type=int, default=30,
                        help="目标帧率 (默认: 30)")
    parser.add_argument("--no-preview", action="store_true",
                        help="禁用 OpenCV 预览窗口 (后台模式)")
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    print("=" * 50)
    print("Camera → NDI 基础转发")
    print("=" * 50)

    app = CameraToNDI(
        camera_id=args.camera,
        ndi_name=args.ndi_output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        show_preview=not args.no_preview,
    )
    app.run()


if __name__ == "__main__":
    main()
