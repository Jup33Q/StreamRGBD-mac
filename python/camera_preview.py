#!/usr/bin/env python3
"""
Camera Preview Worker — converts raw RGB frames to JPEG for live preview.

Reads length-prefixed RGB frames from stdin, writes length-prefixed JPEG
frames to stdout. Designed to be spawned via Erlang Port from Elixir.

Wire protocol (same as inference_worker.py):
  Input:  <<width::32-little, height::32-little, rgb::binary>>
  Output: <<jpeg_size::32-little, jpeg::binary>>
"""
import os
import sys
import struct
import signal

import numpy as np
import cv2


def read_exact(n):
    """Read exactly n bytes from stdin, returning None on EOF."""
    data = b""
    while len(data) < n:
        chunk = sys.stdin.buffer.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def read_frame():
    """Read one RGB frame."""
    header = read_exact(8)
    if header is None:
        return None
    width, height = struct.unpack("<II", header)
    pixel_count = width * height * 3
    pixels = read_exact(pixel_count)
    if pixels is None:
        return None
    return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)


def write_frame(frame_rgb, quality=85):
    """Encode RGB frame to JPEG and write length-prefixed to stdout."""
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        return False
    jpeg = encoded.tobytes()
    sys.stdout.buffer.write(struct.pack("<I", len(jpeg)))
    sys.stdout.buffer.write(jpeg)
    sys.stdout.buffer.flush()
    return True


def main():
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

    print(f"Camera preview worker PID {os.getpid()} starting", file=sys.stderr, flush=True)

    # Emit ready packet: jpeg_size=0 signals readiness, carries PID and TID
    sys.stdout.buffer.write(
        struct.pack("<III", 0, os.getpid(), 0)
    )
    sys.stdout.buffer.flush()

    while True:
        frame = read_frame()
        if frame is None:
            print("STDIN closed, exiting", flush=True)
            break
        write_frame(frame)


if __name__ == "__main__":
    main()
