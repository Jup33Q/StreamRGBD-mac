#!/usr/bin/env python3
"""NDI helper utilities shared across camera/NDI apps."""
import numpy as np


def _check_ndi():
    """Return True if the NDIlib Python bindings are available."""
    try:
        import NDIlib
        return True
    except Exception:
        return False


def _create_ndi_sender(name):
    """Create and return an NDI sender with the given source name."""
    import NDIlib

    send_settings = NDIlib.SendCreate(p_ndi_name=name)
    sender = NDIlib.send_create(send_settings)
    if not sender:
        print(f"WARNING: Failed to create NDI sender: '{name}'")
        return None
    print(f"NDI sender created: '{name}'")
    return sender


def _bgr_to_ndi(frame_bgr):
    """Convert an OpenCV BGR frame to an NDI BGRX video frame."""
    import NDIlib

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


def _send_ndi(sender, frame_bgr):
    """Send a BGR frame through the given NDI sender."""
    if not sender:
        return
    try:
        import NDIlib
        ndi_frame = _bgr_to_ndi(frame_bgr)
        NDIlib.send_send_video_v2(sender, ndi_frame)
    except Exception as e:
        print(f"NDI send error: {e}")
