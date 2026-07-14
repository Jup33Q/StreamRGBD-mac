# StreamRGBD-mac

Real-time camera / NDI image-to-image transformation with optional RGBD depth output
on Apple Silicon, accelerated with CoreML. Controlled via an HTTP API, an Elixir
GenServer, and a Phoenix web dashboard.

**22.7 FPS** at 512x512 resolution on Apple M3 Ultra with SDXS-512.

## Requirements

- **macOS 14+** (Sonoma or later)
- **Apple Silicon** (M1 / M2 / M3 / M4 series)
- **Python 3.9-3.12** (coremltools does not support 3.13+)
- Camera (built-in or USB webcam)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Jup33Q/StreamRGBD-mac.git
cd StreamRGBD-mac

# 2. Setup environment
chmod +x python/setup.sh
python/setup.sh

# 3. Activate
source .venv/bin/activate

# 4. Convert models to CoreML (one-time, ~5 minutes)
python python/scripts/convert_models.py

# 5. Run camera
python python/camera.py --prompt "oil painting style, masterpiece"
```

## Project Layout

```
.
├── python/                       # Python StreamDiffusion + RGBD pipeline
│   ├── camera.py
│   ├── camera_rgbd.py
│   ├── camera_ndi.py
│   ├── streamdiffusion_api.py   # Flask HTTP API for remote control
│   ├── requirements.txt
│   └── setup.sh
├── phx/                          # Elixir Phoenix web UI (SQLite3)
│   └── lib/streamdiffusion_mac/
│       ├── stream_rgbd.ex       # GenServer that owns the Python API process
│       ├── pipeline_agent.ex
│       └── ...
├── coreml_models/               # Converted CoreML models (generated)
└── README.md
```

## Dependencies

`python/setup.sh` installs all dependencies from `python/requirements.txt`. Key version constraints:

| Package | Default | Legacy (`--legacy`) |
|---------|---------|---------------------|
| PyTorch | 2.1-2.5.x | 2.0.x |
| CoreML Tools | 7.x-8.x | 6.x |
| diffusers | 0.21+ | 0.21.x |
| numpy | 1.x (< 2.0) | 1.x (< 2.0) |

- **Default** (`requirements.txt`): Tested with torch 2.5.1 + coremltools 8.3 on M3 Ultra / M4 Max.
- **Legacy** (`requirements-legacy.txt`): Conservative versions for environments where the default fails.

### Troubleshooting

If `python/setup.sh` fails or you experience issues, try the legacy profile:

```bash
rm -rf .venv
python/setup.sh --legacy
```

## Manual Installation

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r python/requirements.txt

# Convert models
python python/scripts/convert_models.py

# Run
python python/camera.py
```

## Usage

### Basic

```bash
# Default (SDXS-512, best speed/quality balance)
python python/camera.py --prompt "oil painting style, masterpiece"

# Watercolor style
python python/camera.py --prompt "watercolor painting, soft brushstrokes"

# With built-in prompt gallery (10 styles, press n/p to switch)
python python/camera.py --prompts
```

### Camera Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save current frame |
| `n` / `p` | Next / Previous prompt |
| `+` / `-` | Adjust camera blend ratio |
| `e` / `d` | Adjust EMA smoothing |

### Advanced Options

```bash
# Use SD-Turbo instead of SDXS (slower but different style)
python python/camera.py --model sd-turbo --prompt "anime style"

# Blend camera with AI output (30% camera)
python python/camera.py --blend 0.3

# Adjust temporal smoothing
python python/camera.py --ema 0.9 --feedback 0.4

# Lower resolution for faster inference on smaller Macs
python python/camera.py --render-size 384

# Select camera device
python python/camera.py --camera 1
```

### Model Conversion

```bash
# Convert SDXS-512 (default, recommended)
python python/scripts/convert_models.py

# Convert SD-Turbo
python python/scripts/convert_models.py --model sd-turbo

# Custom output directory
python python/scripts/convert_models.py --output-dir ./my_models
python python/camera.py --coreml-dir ./my_models
```

## RGBD Output (Depth Anything)

`camera_rgbd.py` extends the pipeline to produce a 4-channel RGBD frame:
StreamDiffusion's AI output is used as the RGB image, Depth Anything estimates
depth, and the two are concatenated into `H x W x 4` uint8 (RGB + depth as alpha).

### Backends (auto-selected)

1. **CoreML DA3-Small** (recommended on macOS / Apple Silicon) — load a pre-converted `.mlpackage`.
2. **PyTorch DA3-Small** — requires the official `depth_anything_3` package.
3. **PyTorch DA2-Small** — Hugging Face Transformers pipeline, works out-of-the-box on macOS.

### Usage

```bash
# Default: auto-selects CoreML -> DA3 PyTorch -> DA2 PyTorch
python python/camera_rgbd.py --prompt "oil painting style, masterpiece"

# Force the Transformers V2 model (no extra dependencies)
python python/camera_rgbd.py --depth-backend pytorch --depth-model da2-small --prompt "watercolor"
```

Press `s` while running to save `capture_rgbd_<timestamp>.png` (RGBA) and a side-by-side visualization.

```bash
# Preview depth as a grayscale alpha overlay
python python/camera_rgbd.py --depth-preview-mode alpha --prompt "oil painting style"

# Preview depth as a colored alpha overlay
python python/camera_rgbd.py --depth-preview-mode alpha_color --prompt "oil painting style"

# Cycle preview modes at runtime by pressing `m`
```

### Converting DA3-Small to CoreML

The official Depth Anything 3 package is not yet easy to install on macOS, but a
community CoreML converter is available:

```bash
# 1. Clone the converter and the DA3-SMALL weights
git clone https://github.com/LSQzzx/Depth-Anything-3-for-CoreML.git
cd Depth-Anything-3-for-CoreML

# 2. Install uv and git-lfs if needed
brew install uv git-lfs
git lfs install

# 3. Download weights
git clone https://huggingface.co/depth-anything/DA3-SMALL

# 4. Convert
uv sync
uv run coreml_converter/convert2coreml.py

# 5. Copy the result into this project
cp da3.mlpackage /path/to/StreamRGBD-mac/coreml_models/da3_small.mlpackage
```

Then run `python python/camera_rgbd.py` and it will load the CoreML depth model automatically.

## Phoenix Web UI & Video Stream

A Phoenix project lives in `phx/`. The dashboard now uses Membrane to capture
the local camera, spawns a Python inference worker as a CLI command, and serves
the processed video stream to the browser via MJPEG over HTTP.

### Components

- `StreamdiffusionMac.CameraPipeline` — Membrane pipeline that captures camera
  frames and converts them to RGB.
- `StreamdiffusionMac.InferenceWorker` — GenServer that owns the Python
  `inference_worker.py` Port and forwards frames back and forth.
- `StreamdiffusionMac.VideoStreamer` — GenServer that holds the latest
  processed JPEG frame.
- `StreamdiffusionMac.StreamRGBD` — orchestrator that starts/stops the pipeline
  and worker.
- Web dashboard at `/` with a live video preview.
- JSON endpoints under `/api/stream/*` and MJPEG stream at `/api/stream/video`.

### Setup

```bash
cd phx
mix deps.get
mix ecto.setup
```

Requires FFmpeg for Membrane camera capture:

```bash
brew install ffmpeg
```

### Run

```bash
cd phx
mix phx.server
```

Then open <http://localhost:4000> and click **Start Engine**.

### Programmatic control (IEx)

```elixir
StreamdiffusionMac.StreamRGBD.start_engine(prompt: "oil painting style, masterpiece")
StreamdiffusionMac.StreamRGBD.set_prompt("cyberpunk city, neon lights")
StreamdiffusionMac.StreamRGBD.status()
StreamdiffusionMac.StreamRGBD.stop_engine()
```

### JSON API

| Method | Path | Body |
|--------|------|------|
| `POST` | `/api/stream/start` | `{"prompt":"...", "model":"sdxs", "render-size":512, "output-size":512, "width":640, "height":480}` |
| `POST` | `/api/stream/stop` | — |
| `POST` | `/api/stream/prompt` | `{"prompt":"..."}` |
| `GET`  | `/api/stream/status` | — |
| `GET`  | `/api/stream/video` | — |

### Examples

```bash
# Start the engine
curl -X POST http://127.0.0.1:4000/api/stream/start \
  -H "Content-Type: application/json" \
  -d '{"prompt":"oil painting style, masterpiece"}'

# Change prompt
curl -X POST http://127.0.0.1:4000/api/stream/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"watercolor painting"}'

# Read status
curl http://127.0.0.1:4000/api/stream/status

# MJPEG stream (open in browser or save)
curl http://127.0.0.1:4000/api/stream/video
```

### Useful tasks

```bash
# Reset the SQLite database
cd phx && mix ecto.reset

# Run tests
cd phx && mix test

# Run precommit checks
cd phx && mix precommit
```

## Architecture

```
Camera Thread ──→ latest_frame ──→ Inference Thread ──→ latest_ai_result
     (30 FPS)         │              (CoreML pipeline)       │
                      │                                      │
                      └──────→ Display Thread ←──────────────┘
                               (blends camera + AI, 30+ FPS)
```

The pipeline decouples camera capture, AI inference, and display into three independent threads. The inference thread runs the full CoreML pipeline on every frame:

1. **Preprocess**: Center crop, resize to 512x512, normalize
2. **VAE Encode**: Image → Latent space (CoreML TAESD, ~5ms)
3. **Noise Addition**: Fixed noise for temporal coherence
4. **UNet Inference**: Single-step denoising (CoreML, ~24ms for SDXS)
5. **VAE Decode**: Latent → Image (CoreML TAESD, ~5ms)
6. **Postprocess**: Denormalize, resize, display

Temporal coherence is maintained through:
- **Fixed noise seed**: Same noise pattern every frame eliminates flickering
- **Latent feedback**: 30% of previous frame's denoised latent blended into current input
- **EMA smoothing**: Exponential moving average on display output

## Performance

Benchmarked on Apple M3 Ultra (60-core GPU, 512GB unified memory):

| Model | Parameters | UNet Latency | Camera FPS | Quality |
|-------|-----------|-------------|------------|---------|
| **SDXS-512** | **328M** | **24.4ms** | **22.7** | Good |
| SD-Turbo | 866M | 53.2ms | 13.8 | Good |
| Tiny-SD | 323M | 31.3ms | ~20 | Fair |

> Performance scales with GPU core count. Expected approximate FPS:
> - M1/M2: ~5-8 FPS
> - M1/M2 Pro: ~8-12 FPS
> - M1/M2/M3 Max: ~12-18 FPS
> - **M4 Max (40-core GPU, 128GB): ~15.4 FPS** (SDXS-512, measured)
> - M3 Ultra: ~22 FPS

## Experiment Report

This project is the result of a systematic 10-phase optimization study on real-time diffusion model inference on Apple Silicon. Below is a summary of key findings.

### What Works on Apple Silicon

| Technique | Effect | Notes |
|-----------|--------|-------|
| **CoreML conversion** | **+64%** | Only effective UNet acceleration method |
| **Distilled models (SDXS)** | **+118%** | Best speed/quality trade-off |
| 3-thread pipeline | Smooth display | Decouples inference from rendering |

### What Does NOT Work on Apple Silicon

| Technique | Effect | Why |
|-----------|--------|-----|
| Quantization (INT8 to 2-bit) | 0% | M3 Ultra is compute-bound, not memory-bandwidth-bound |
| Token Merging (ToMe) | -10% | MPS overhead exceeds attention savings |
| Parallel CoreML inference | 0% | Metal serializes GPU commands |
| Neural Engine for UNet | -19% to -520% | ANE unsuitable for large (866M) models |
| torch.compile | Crash | MPS backend not supported |
| Attention Slicing | -40% | MPS memory management overhead |

### Key Insight: CUDA Optimization Wisdom Does Not Apply

The most important finding is that optimization techniques established for NVIDIA GPUs and the CUDA ecosystem largely **do not transfer** to Apple Silicon's unified memory architecture:

- **Quantization is ineffective** because Apple Silicon is compute-bound (not memory-bandwidth-bound). The 800 GB/s unified memory bandwidth is sufficient for model weights, so reducing precision doesn't help.
- **Parallel inference is impossible** because CoreML serializes Metal GPU commands, unlike CUDA Streams which allow fine-grained kernel-level parallelism.
- **The software ecosystem is immature** compared to CUDA's decades of optimization (cuDNN, TensorRT, xformers, Flash Attention). torch.compile doesn't work on MPS, and many PyTorch operations have suboptimal Metal implementations.

### Negative Results

Several creative approaches were also tested and yielded negative results:

- **kNN search-based synthesis** (Phase 7): 512GB memory enables searching 100M vectors in 0.5ms, but kNN retrieval fundamentally cannot replace the continuous nonlinear function approximation of a UNet.
- **pix2pix-turbo** (Phase 8): Skip-connection VAE design prevents CoreML conversion, creating a 160ms VAE bottleneck (vs 53ms UNet). Result: 4 FPS.
- **Optical flow frame skipping** (Phase 9): Warping between UNet frames produces jelly-like distortion artifacts. 17.4 FPS, worse than SDXS baseline.
- **Knowledge distillation** (Phase 10): 875K-parameter feedforward CNN trained with L1 loss produces blank output. The capacity gap vs 328M-parameter diffusion model is too large.

### Full Paper

A detailed academic paper covering all experiments is available:
- [paper.tex](paper.tex) (Japanese)
- [paper_en.tex](paper_en.tex) (English)

## License

MIT License

## Citation

```bibtex
@article{ochiai2025streamdiffusion_mac,
  title={Systematic Optimization of Real-Time Diffusion Model Inference on Apple M3 Ultra},
  author={Ochiai, Yoichi},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgments

- [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) — Original pipeline architecture
- [SDXS](https://github.com/IDKiro/sdxs) — Distillation-specialized model
- [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) — One-step diffusion baseline
- [TAESD](https://github.com/madebyollin/taesd) — Tiny Autoencoder for Stable Diffusion
