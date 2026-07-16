#!/usr/bin/env python3
"""
StreamDiffusion for Mac — HTTP API Server

Exposes the camera/NDI/RGBD pipeline over a local HTTP API so that the
Phoenix Elixir backend (or any other client) can:

  - start / stop the engine
  - change the prompt
  - switch between camera and NDI input
  - set the NDI input source name
  - set the NDI output source name
  - read runtime status

The server runs on 127.0.0.1 by default and keeps the heavy CoreML/NDI
runtime isolated from the BEAM VM.

Endpoints (all JSON):

  POST /start
      { "mode": "camera|ndi", "prompt": "...", "ndi_source": "...",
        "ndi_output": "...", "depth": false, ... }
      Optional keys mirror the CLI arguments. Returns current status.

  POST /stop
      Stops capture/inference threads and releases resources.

  POST /prompt
      { "prompt": "..." }

  POST /input_mode
      { "mode": "camera|ndi" }

  POST /ndi_input
      { "source": "..." }

  POST /ndi_output
      { "name": "..." }

  GET /status
      Returns { "running", "mode", "prompt", "ndi_source", "ndi_output",
                "ai_fps", "frame_count", "error" }.
"""
import os
import sys
import time
import json
import signal
import argparse
import threading
import traceback
from functools import wraps

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipelines.coreml import Pipeline, COREML_DIR
from configs import DEFAULT_PROMPTS, MODEL_CONFIGS


def _patch_pipeline_set_prompt():
    """Add a runtime prompt setter to the existing Pipeline class."""
    if hasattr(Pipeline, "set_prompt"):
        return

    def set_prompt(self, prompt):
        """Encode a new prompt and make it the active target embedding."""
        import torch
        with torch.no_grad():
            ti = self._tokenizer(
                prompt,
                padding="max_length",
                max_length=self._tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            embeds = (
                self._text_encoder(ti.input_ids.to("mps"))[0]
                .cpu()
                .to(torch.float16)
                .numpy()
            )
        self._all_prompts = [prompt]
        self._all_embeds = [embeds]
        self._prompt_index = 0
        self._target_embeds = embeds
        self._current_prompt = prompt
        return prompt

    Pipeline.set_prompt = set_prompt


_patch_pipeline_set_prompt()


def _json_response(data, status_code=200):
    from flask import Response
    return Response(
        json.dumps(data, ensure_ascii=False),
        status=status_code,
        mimetype="application/json",
    )


def _require_running(manager):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not manager.is_running():
                return _json_response({"error": "engine not running"}, 409)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


class CameraCapture:
    """Threaded OpenCV camera capture."""

    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        # Warmup
        for _ in range(30):
            ret, _ = self.cap.read()
            if ret:
                break
            time.sleep(0.05)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._latest_frame = frame

    def get_frame(self):
        with self._lock:
            return self._latest_frame

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None
        self._latest_frame = None


class NDIInputCapture:
    """Threaded NDI source capture."""

    FIND_TIMEOUT_MS = 10000
    CAPTURE_TIMEOUT_MS = 100

    def __init__(self, source_name=None):
        self.source_name = source_name
        self._receiver = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame = None
        self._ndi_available = self._check_ndi()

    @staticmethod
    def _check_ndi():
        try:
            import NDIlib
            return True
        except Exception:
            return False

    def _find_source(self):
        import NDIlib

        if not NDIlib.initialize():
            raise RuntimeError("NDI initialization failed")

        finder = NDIlib.find_create_v2()
        if not finder:
            raise RuntimeError("Failed to create NDI finder")

        sources = []
        waited = 0
        step = 500
        while waited < self.FIND_TIMEOUT_MS:
            NDIlib.find_wait_for_sources(finder, step)
            waited += step
            sources = NDIlib.find_get_current_sources(finder)
            if sources:
                break

        if not sources:
            NDIlib.find_destroy(finder)
            raise RuntimeError("No NDI sources found")

        selected = sources[0]
        if self.source_name:
            for s in sources:
                if self.source_name.lower() in s.ndi_name.lower():
                    selected = s
                    break

        name = selected.ndi_name
        NDIlib.find_destroy(finder)
        return selected, name

    def _create_receiver(self, source):
        import NDIlib

        recv_settings = NDIlib.RecvCreateV3()
        recv_settings.source_to_connect_to = source
        recv_settings.color_format = NDIlib.RecvColorFormat.RECV_COLOR_FORMAT_BGRX_BGRA
        recv_settings.bandwidth = NDIlib.RecvBandwidth.RECV_BANDWIDTH_HIGHEST
        recv_settings.allow_video_fields = False

        receiver = NDIlib.recv_create_v3(recv_settings)
        if not receiver:
            raise RuntimeError("Failed to create NDI receiver")
        return receiver

    @staticmethod
    def _ndi_to_bgr(video_frame):
        if video_frame.data is None or video_frame.data.size == 0:
            return None

        h = video_frame.yres
        w = video_frame.xres
        if h <= 0 or w <= 0:
            return None

        if video_frame.data.ndim == 1:
            stride = video_frame.line_stride_in_bytes or w * 4
            expected = h * stride
            if video_frame.data.size < expected:
                return None
            flat = video_frame.data[:expected]
            img = flat.reshape(h, stride)
            if stride > w * 4:
                img = img[:, :w * 4]
            img = img.reshape(h, w, 4)
        elif video_frame.data.ndim == 2:
            img = video_frame.data.reshape(h, -1)
            if img.shape[1] >= w * 4:
                img = img[:, :w * 4].reshape(h, w, 4)
            else:
                return None
        elif video_frame.data.ndim == 3:
            img = video_frame.data
            if img.shape[2] < 3:
                return None
        else:
            return None

        if img.shape[2] >= 3:
            return img[:, :, :3].copy()
        return None

    def start(self):
        if not self._ndi_available:
            raise RuntimeError("NDIlib is not installed")

        source, name = self._find_source()
        self._receiver = self._create_receiver(source)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return name

    def _loop(self):
        import NDIlib

        while self._running:
            frame_type, video_frame, _, _ = NDIlib.recv_capture_v3(
                self._receiver, timeout_in_ms=self.CAPTURE_TIMEOUT_MS
            )
            if frame_type == NDIlib.FrameType.FRAME_TYPE_VIDEO:
                frame = self._ndi_to_bgr(video_frame)
                if frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                try:
                    NDIlib.recv_free_video_v2(self._receiver, video_frame)
                except Exception:
                    pass
            elif frame_type == NDIlib.FrameType.FRAME_TYPE_ERROR:
                break

    def get_frame(self):
        with self._lock:
            return self._latest_frame

    def stop(self):
        import NDIlib

        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._receiver:
            try:
                NDIlib.recv_destroy(self._receiver)
            except Exception:
                pass
            self._receiver = None
        self._latest_frame = None


class NDIOutput:
    """Wraps an NDI sender."""

    def __init__(self, output_name="StreamDiffusion-Mac"):
        self.output_name = output_name
        self._sender = None
        self._ndi_available = self._check_ndi()

    @staticmethod
    def _check_ndi():
        try:
            import NDIlib
            return True
        except Exception:
            return False

    def start(self):
        if not self._ndi_available:
            raise RuntimeError("NDIlib is not installed")

        import NDIlib

        send_settings = NDIlib.SendCreate(p_ndi_name=self.output_name)
        self._sender = NDIlib.send_create(send_settings)
        if not self._sender:
            raise RuntimeError("Failed to create NDI sender")
        return self.output_name

    @staticmethod
    def _bgr_to_ndi(frame_bgr):
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

    def send(self, frame_bgr):
        if self._sender is None:
            return
        import NDIlib

        try:
            ndi_frame = self._bgr_to_ndi(frame_bgr)
            NDIlib.send_send_video_v2(self._sender, ndi_frame)
        except Exception as e:
            print(f"NDI send error: {e}")

    def stop(self):
        if self._sender:
            import NDIlib

            try:
                NDIlib.send_destroy(self._sender)
            except Exception:
                pass
            self._sender = None


class StreamDiffusionManager:
    """Owns the pipeline, capture, inference and optional NDI output threads."""

    VALID_MODES = {"camera", "ndi"}

    def __init__(self):
        self._pipeline = None
        self._capture = None
        self._ndi_output = None
        self._inference_thread = None
        self._send_thread = None
        self._running = False
        self._shutdown = False

        self._config = {
            "mode": "camera",
            "prompt": "oil painting style, masterpiece, highly detailed",
            "prompts": False,
            "model": "sdxs",
            "render_size": 512,
            "output_size": 512,
            "strength": 0.5,
            "blend": 0.0,
            "ema": 0.4,
            "feedback": 0.1,
            "camera_id": 0,
            "ndi_source": None,
            "ndi_output": "StreamDiffusion-Mac",
            "depth": False,
            "depth_model": "auto",
            "depth_backend": "auto",
            "depth_coreml_path": None,
            "coreml_dir": COREML_DIR,
        }

        self._state_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._latest_ai_result = None
        self._latest_frame = None
        self._ai_fps = 0.0
        self._ai_count = 0
        self._send_count = 0
        self._error = None

    def _set_error(self, msg):
        with self._state_lock:
            self._error = msg
        print(f"ERROR: {msg}")

    def _clear_error(self):
        with self._state_lock:
            self._error = None

    def is_running(self):
        with self._state_lock:
            return self._running

    def status(self):
        with self._state_lock:
            cfg = self._config.copy()
            err = self._error
            running = self._running
        with self._result_lock:
            fps = self._ai_fps
            count = self._ai_count
            send_count = self._send_count
        return {
            "running": running,
            "mode": cfg["mode"],
            "prompt": cfg["prompt"],
            "ndi_source": cfg["ndi_source"],
            "ndi_output": cfg["ndi_output"],
            "model": cfg["model"],
            "render_size": cfg["render_size"],
            "ai_fps": round(fps, 2),
            "frame_count": count,
            "ndi_send_count": send_count,
            "error": err,
        }

    def start(self, overrides=None):
        with self._state_lock:
            if self._running:
                return self.status()
            if overrides:
                self._config.update({k: v for k, v in overrides.items() if k in self._config})
            cfg = self._config.copy()

        try:
            self._clear_error()
            self._build_pipeline(cfg)
            self._start_capture(cfg)
            self._start_inference()
            if cfg["mode"] == "ndi" or cfg.get("ndi_output"):
                self._start_ndi_output(cfg)

            with self._state_lock:
                self._running = True
        except Exception as e:
            self._set_error(str(e))
            self.stop()
            raise

        return self.status()

    def stop(self):
        with self._state_lock:
            self._running = False

        self._shutdown = True

        if self._send_thread:
            self._send_thread.join(timeout=2)
            self._send_thread = None
        if self._ndi_output:
            self._ndi_output.stop()
            self._ndi_output = None

        if self._inference_thread:
            self._inference_thread.join(timeout=2)
            self._inference_thread = None

        if self._capture:
            self._capture.stop()
            self._capture = None

        self._pipeline = None
        with self._result_lock:
            self._latest_ai_result = None
            self._latest_frame = None
            self._ai_fps = 0.0
            self._ai_count = 0
            self._send_count = 0

        return self.status()

    def _build_pipeline(self, cfg):
        prompts = DEFAULT_PROMPTS if cfg["prompts"] else None
        kwargs = {
            "model_name": cfg["model"],
            "render_size": cfg["render_size"],
            "output_size": cfg["output_size"],
            "prompt": cfg["prompt"],
            "strength": cfg["strength"],
            "prompts": prompts,
            "latent_feedback": cfg["feedback"],
            "coreml_dir": cfg["coreml_dir"],
        }

        if cfg["depth"]:
            from camera_rgbd import RGBDPipeline

            kwargs.update(
                {
                    "depth_model": cfg["depth_model"],
                    "depth_backend": cfg["depth_backend"],
                    "depth_coreml_path": cfg["depth_coreml_path"],
                }
            )
            self._pipeline = RGBDPipeline(**kwargs)
        else:
            self._pipeline = Pipeline(**kwargs)

    def _start_capture(self, cfg):
        mode = cfg["mode"]
        if mode == "camera":
            self._capture = CameraCapture(camera_id=cfg["camera_id"])
        elif mode == "ndi":
            self._capture = NDIInputCapture(source_name=cfg["ndi_source"])
        else:
            raise ValueError(f"Unknown input mode: {mode}")
        self._capture.start()

    def _start_inference(self):
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    def _inference_loop(self):
        fps_times = []
        while not self._shutdown:
            frame = self._capture.get_frame() if self._capture else None
            if frame is None:
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()
            try:
                result = self._pipeline.process_frame(frame)
            except Exception as e:
                self._set_error(f"inference failed: {e}")
                time.sleep(0.1)
                continue

            elapsed = time.perf_counter() - t0
            fps_times.append(elapsed)
            if len(fps_times) > 30:
                fps_times.pop(0)

            with self._result_lock:
                self._latest_ai_result = result
                self._latest_frame = frame
                self._ai_count += 1
                self._ai_fps = 1.0 / (sum(fps_times) / len(fps_times))

    def _start_ndi_output(self, cfg):
        self._ndi_output = NDIOutput(output_name=cfg["ndi_output"])
        self._ndi_output.start()
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._send_thread.start()

    def _send_loop(self):
        while not self._shutdown:
            with self._result_lock:
                result = self._latest_ai_result
            if result is None:
                time.sleep(0.001)
                continue

            if self._ndi_output:
                self._ndi_output.send(result)
                with self._result_lock:
                    self._send_count += 1
            time.sleep(1 / 30.0)

    def set_prompt(self, prompt):
        with self._state_lock:
            self._config["prompt"] = prompt
            pipeline = self._pipeline
        if pipeline is not None:
            pipeline.set_prompt(prompt)
        return self.status()

    def set_input_mode(self, mode):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}")
        with self._state_lock:
            old_mode = self._config["mode"]
            self._config["mode"] = mode
            cfg = self._config.copy()
            running = self._running

        if running and mode != old_mode:
            # Restart capture while keeping the pipeline alive.
            if self._capture:
                self._capture.stop()
            self._start_capture(cfg)
        return self.status()

    def set_ndi_input(self, source_name):
        with self._state_lock:
            self._config["ndi_source"] = source_name
            mode = self._config["mode"]
            running = self._running

        if running and mode == "ndi":
            if self._capture:
                self._capture.stop()
            self._start_capture(self._config.copy())
        return self.status()

    def set_ndi_output(self, output_name):
        with self._state_lock:
            self._config["ndi_output"] = output_name
            running = self._running

        if running:
            if self._ndi_output:
                self._ndi_output.stop()
            self._ndi_output = NDIOutput(output_name=output_name)
            self._ndi_output.start()
        return self.status()


class APIServer:
    def __init__(self, host="127.0.0.1", port=8787):
        self.host = host
        self.port = port
        self.manager = StreamDiffusionManager()
        self.app = self._create_app()

    def _create_app(self):
        from flask import Flask, request

        app = Flask(__name__)
        manager = self.manager
        require = _require_running(manager)

        @app.get("/")
        def index():
            return _json_response({"service": "streamdiffusion-mac-api", "status": "/status"})

        @app.post("/start")
        def start():
            try:
                payload = request.get_json(silent=True) or {}
                status = manager.start(payload)
                return _json_response(status)
            except Exception as e:
                traceback.print_exc()
                return _json_response({"error": str(e)}, 500)

        @app.post("/stop")
        def stop():
            try:
                return _json_response(manager.stop())
            except Exception as e:
                traceback.print_exc()
                return _json_response({"error": str(e)}, 500)

        @app.post("/prompt")
        @require
        def prompt():
            payload = request.get_json(silent=True) or {}
            prompt_text = payload.get("prompt")
            if not prompt_text or not isinstance(prompt_text, str):
                return _json_response({"error": "missing or invalid 'prompt'"}, 400)
            return _json_response(manager.set_prompt(prompt_text))

        @app.post("/input_mode")
        @require
        def input_mode():
            payload = request.get_json(silent=True) or {}
            mode = payload.get("mode")
            try:
                return _json_response(manager.set_input_mode(mode))
            except ValueError as e:
                return _json_response({"error": str(e)}, 400)

        @app.post("/ndi_input")
        @require
        def ndi_input():
            payload = request.get_json(silent=True) or {}
            source = payload.get("source")
            if source is None:
                return _json_response({"error": "missing 'source'"}, 400)
            return _json_response(manager.set_ndi_input(source))

        @app.post("/ndi_output")
        @require
        def ndi_output():
            payload = request.get_json(silent=True) or {}
            name = payload.get("name")
            if name is None:
                return _json_response({"error": "missing 'name'"}, 400)
            return _json_response(manager.set_ndi_output(name))

        @app.get("/status")
        def status():
            return _json_response(manager.status())

        return app

    def run(self):
        # Print a ready marker so the parent process knows the HTTP server is up.
        # Also emit the OS PID and main thread ID so the parent can identify this
        # StreamDiffusion instance.
        def _ready():
            time.sleep(0.5)
            print(
                f"STREAMDIFFUSION_INSTANCE_PID {os.getpid()}", flush=True
            )
            print(
                f"STREAMDIFFUSION_INSTANCE_TID {threading.current_thread().ident}",
                flush=True,
            )
            print(f"STREAMDIFFUSION_API_READY http://{self.host}:{self.port}", flush=True)

        threading.Thread(target=_ready, daemon=True).start()
        self.app.run(host=self.host, port=self.port, threaded=True, debug=False, use_reloader=False)


def _watch_stdin_and_exit():
    """Exit when the controlling port closes stdin."""
    def _loop():
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
        except Exception:
            pass
        print("STDIN closed, exiting...", flush=True)
        os._exit(0)

    threading.Thread(target=_loop, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="StreamDiffusion for Mac — HTTP API")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

    _watch_stdin_and_exit()

    print(f"StreamDiffusion API starting on {args.host}:{args.port}", flush=True)
    server = APIServer(host=args.host, port=args.port)
    server.run()


if __name__ == "__main__":
    main()
