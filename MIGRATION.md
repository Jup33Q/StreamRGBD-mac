# Migration Guide: v2.0 → v2.1

## Summary

Removed the Phoenix/Elixir web frontend. The project is now a **pure Python command-line tool** using OpenCV for real-time display.

---

## Why the migration?

1. **macOS camera permissions** are restricted to GUI apps (Terminal.app); the Phoenix backend running as a background process could not reliably request camera access from macOS TCC.
2. **Reduced dependencies**: No Elixir, Phoenix, Membrane, FFmpeg, or SQLite required.
3. **Simpler deployment**: Single `start_streamrgbd.sh` script, no compilation or build step.
4. **Lower latency**: Direct Python `cv2.imshow()` window eliminates HTTP/WebSocket and MJPEG streaming overhead.

---

## What was removed?

| Removed Component | Technology | Replaced By |
|-------------------|-----------|-------------|
| `phx/` directory | Phoenix 1.7 + LiveView 1.0 + HEEx | Direct `python/camera_rgbd.py` |
| `StreamdiffusionMac.StreamRGBD` | Elixir GenServer orchestrator | `RGBDCameraApp` Python class |
| `StreamdiffusionMac.InferenceWorker` | Erlang Port (`Port.open`) | Python `threading.Thread` |
| `StreamdiffusionMac.CameraPipeline` | Membrane Core 1.3 + `membrane_camera_capture_plugin` | `cv2.VideoCapture(0)` |
| `StreamdiffusionMac.VideoStreamer` | GenServer JPEG frame store | `threading.Lock` + shared `np.ndarray` |
| MJPEG HTTP stream | `Plug.Conn.chunk` / `multipart/x-mixed-replace` | `cv2.imshow()` + `cv2.waitKey()` |
| PubSub status broadcast | `Phoenix.PubSub.broadcast` | Terminal stdout (`print`) |
| SQLite3 database | `ecto_sqlite3` | Not needed (no persistence) |
| Elixir wire protocol | `<<width::32-little, height::32-little, rgb::binary>>` | Direct Python memory buffers |

---

## What remains unchanged?

- **CoreML inference pipeline**: `Pipeline` and `RGBDPipeline` classes (`python/camera.py`, `python/camera_rgbd.py`)
- **VAE Encoder/Decoder**: TAESD CoreML models (`taesd_encoder_512.mlpackage`, `taesd_decoder.mlpackage`)
- **UNet inference**: SDXS-512 / SD-Turbo CoreML models (`unet_sdxs_512.mlpackage`)
- **Temporal coherence**: Fixed noise seed, latent feedback, EMA smoothing
- **Depth estimation**: Depth Anything V2/V3 backends (PyTorch / CoreML)
- **Model conversion scripts**: `python/scripts/convert_models.py`

---

## New tech stack (v2.1)

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Runtime** | Python | 3.9–3.12 | Inference backend |
| **ML Framework** | PyTorch | 2.1–2.5.x | Text encoder, MPS device management |
| **Diffusion** | diffusers | 0.21+ | Stable Diffusion pipeline components |
| **Apple Acceleration** | coremltools | 7.x–8.x | CoreML model conversion & inference |
| **Vision** | OpenCV (cv2) | 4.x | Camera capture, preprocessing, display |
| **Numeric** | NumPy | 1.x (< 2.0) | Tensor operations, buffer management |
| **Depth** | transformers | 4.x | Depth Anything V2 HuggingFace pipeline |
| **Depth (alt)** | depth_anything_3 | — | Depth Anything V3 (optional, requires clone) |
| **Build** | setuptools | — | Environment setup (distutils shim for coremltools) |

---

## Referenced code & models

The following third-party projects and pretrained models are directly integrated or referenced in the current codebase:

### Pretrained Models

| Model | Source | HuggingFace / GitHub | License |
|-------|--------|---------------------|---------|
| **SDXS-512** | One-step distilled Stable Diffusion | [IDKiro/sdxs-512-0.9](https://huggingface.co/IDKiro/sdxs-512-0.9) | Apache-2.0 |
| **SD-Turbo** | StabilityAI one-step model | [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) | Stability AI Community License |
| **TAESD** | Tiny Autoencoder for SD | [madebyollin/taesd](https://github.com/madebyollin/taesd) | MIT |
| **Depth Anything V2 Small** | Monocular depth estimation | [LiheYoung/Depth-Anything-V2](https://github.com/LiheYoung/Depth-Anything-V2) | Apache-2.0 |
| **Depth Anything V3 Small** | Transformer-based depth | [depth-anything/DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL) | Apache-2.0 |

### Open Source Projects (Architecture & Techniques)

| Project | Component Used | License | URL |
|---------|---------------|---------|-----|
| **StreamDiffusion** | 3-thread pipeline architecture, fixed-noise temporal coherence, latent feedback | MIT | [cumulo-autumn/StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) |
| **Depth-Anything-3-for-CoreML** | CoreML converter for DA3-Small (community) | — | [LSQzzx/Depth-Anything-3-for-CoreML](https://github.com/LSQzzx/Depth-Anything-3-for-CoreML) |

The `camera.py` and `camera_rgbd.py` pipeline implementations are **adaptations** of the StreamDiffusion architecture for Apple Silicon CoreML, not direct forks.

---

## Migration path for existing users

### If you were using the Phoenix web UI (`phx/`)

1. Remove the `phx/` directory (already done in v2.1).
2. Re-run model conversion if needed: `python python/scripts/convert_models.py`
3. Use the new launcher script instead of `mix phx.server`:
   ```bash
   ./start_streamrgbd.sh
   ```
4. Controls are now keyboard-driven in the cv2 window instead of web buttons.

### If you were using `inference_worker.py` standalone

`inference_worker.py` is preserved for Erlang Port compatibility but no longer the primary entry point. Use `camera_rgbd.py` for direct camera input.

---

## File changes in v2.1

```
DELETED:  phx/                          (entire Phoenix project)
MODIFIED: README.md                     (updated badges, tech stack, migration note)
MODIFIED: OPERATION_MANUAL.md           (rewritten for pure Python CLI)
ADDED:    start_streamrgbd.sh           (convenience launcher)
ADDED:    python/scripts/convert_depth_model.py  (DA3 CoreML converter helper)
UNCHANGED: python/camera.py             (core pipeline)
UNCHANGED: python/camera_rgbd.py        (RGBD pipeline)
UNCHANGED: python/inference_worker.py   (Erlang Port compatibility)
UNCHANGED: coreml_models/               (converted models preserved)
```

---

*Migration written: 2025-07-15*
