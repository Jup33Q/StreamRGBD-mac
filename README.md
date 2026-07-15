# StreamRGBD-mac

![Python](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12-blue?logo=python)
![macOS](https://img.shields.io/badge/macOS-14%2B-black?logo=apple)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-orange)
![CoreML](https://img.shields.io/badge/CoreML-Accelerated-green)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS%2FCUDA-red?logo=pytorch)
![FPS](https://img.shields.io/badge/Performance-22.7%20FPS%20%40512%C3%97512-brightgreen)

Real-time camera image-to-image transformation with optional RGBD depth output
on Apple Silicon, accelerated with CoreML.

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

# 5. (Optional) Download LoRAs for style/character control
./start_download_loras.sh

# 6. Run camera
python python/camera.py --prompt "oil painting style, masterpiece"
```

Or use the convenience launcher scripts (after setup):

```bash
# Download all recommended LoRAs (26 style + character models)
./start_download_loras.sh

# RGBD + NDI output (color + depth channels)
./start_streamrgbd.sh

# Camera → NDI forwarding (no AI processing)
./start_camera_to_ndi.sh

# GUI control panel for stream_rgbd
./start_streamrgbd_gui.sh

# NDI source scanner GUI
./start_ndi_scanner.sh
```

## Project Layout

```
.
├── python/                       # Python StreamDiffusion + RGBD pipeline
│   ├── camera.py                 # Basic img2img pipeline (cv2 window)
│   ├── camera_rgbd.py            # RGBD pipeline with depth + NDI output
│   ├── camera_ndi.py             # NDI input support
│   ├── camera_to_ndi.py          # Basic camera → NDI forwarding (no AI)
│   ├── camera_lora.py            # LoRA model support
│   ├── camera_preview.py         # JPEG preview worker (legacy Erlang Port helper)
│   ├── inference_worker.py       # Headless CoreML worker (legacy Erlang Port helper)
│   ├── lkg_bridge.py             # Looking Glass holographic display bridge
│   ├── download_loras.py         # LoRA batch downloader from HuggingFace
│   ├── download_loras_civitai.py # LoRA downloader for Civitai models
│   ├── streamdiffusion_api.py    # Flask HTTP API for remote control
│   ├── ndi_scanner_gui.py        # GUI: scan available NDI sources
│   ├── stream_rgbd_gui.py        # GUI: control panel for stream_rgbd (legacy)
│   ├── stream_rgbd_gui_db.py     # GUI: database-enhanced control panel
│   ├── models.py                 # peewee ORM models (SQLite)
│   ├── db_init.py                # Database initialization & seed data
│   ├── tk_style.py               # Window style module (light/dark theme)
│   ├── DATABASE_GUI_USAGE.md     # Database & GUI integration guide
│   ├── requirements.txt
│   ├── requirements-legacy.txt   # Legacy dependency versions
│   └── setup.sh
│   └── scripts/                  # Model conversion scripts
│       ├── convert_models.py     # Convert SDXS/SD-Turbo to CoreML
│       ├── convert_da3_coreml_v4.py  # Convert DA3-Small to CoreML
│       └── convert_depth_model.py    # Convert depth models to CoreML
├── data/                         # SQLite database & app data
├── coreml_models/               # Converted CoreML models (generated)
├── python/loras/                # Downloaded LoRA models (*.safetensors)
├── start_streamrgbd.sh          # Convenience launcher (RGBD + NDI out)
├── start_streamrgbd_gui.sh      # GUI launcher (database-enhanced)
├── start_camera_to_ndi.sh       # Launcher for camera → NDI forwarding
├── start_download_loras.sh      # Launcher for LoRA batch download
├── start_ndi_scanner.sh         # Launcher for NDI source scanner GUI
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

### Convenience Launcher

```bash
# Start with default settings (SDXS-512, oil painting prompt)
./start_streamrgbd.sh

# Pass custom arguments
./start_streamrgbd.sh --prompt "cyberpunk city, neon lights" --render-size 384
./start_streamrgbd.sh --blend 0.3 --feedback 0.2
```

### Basic img2img

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

A built-in conversion script is provided to generate the CoreML `.mlpackage` from the official DA3-SMALL weights:

```bash
# 1. Ensure DA3-SMALL weights are cached by Hugging Face
#    (they will be downloaded automatically on first use)

# 2. Run the built-in converter
python python/scripts/convert_da3_coreml_v4.py

# 3. The model is saved to coreml_models/da3_small.mlpackage
```

Requirements:
- `depth_anything_3` source must be available at `/tmp/depth-anything-3/src` (clone from [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3))
- DA3-SMALL weights cached at `~/.cache/huggingface/hub/models--depth-anything--DA3-SMALL/...`

Once converted, `camera_rgbd.py --depth-backend auto` will automatically load the CoreML model for the fastest depth inference on Apple Silicon.

> **Note**: The conversion replaces `bicubic` interpolation (unsupported by CoreML) with `bilinear`, and disables multi-view `alt_start` (single-view depth only). These changes do not affect depth estimation quality for single-image inputs.

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

## NDI Output

`camera_rgbd.py` now supports dual NDI output — sending the AI color image and the depth map as two separate NDI sources simultaneously.

```bash
# Default: color → "StreamDiffusion-RGBD-color", depth → "StreamDiffusion-RGBD-depth"
./start_streamrgbd.sh

# Custom NDI source name
./start_streamrgbd.sh --ndi-output "MyProject"
# Generates: "MyProject-color" and "MyProject-depth"
```

### Camera → NDI Forwarding (No AI)

For basic camera capture without AI processing, use `camera_to_ndi.py`:

```bash
./start_camera_to_ndi.sh

# Custom camera, resolution, and NDI name
./start_camera_to_ndi.sh --camera 1 --ndi-output "MyCam" --width 1920 --height 1080

# List available cameras
./start_camera_to_ndi.sh --list-cameras

# Headless mode (no preview window)
./start_camera_to_ndi.sh --no-preview
```

## GUI Tools

### LoRA Downloader

Two download sources are supported: **HuggingFace** (default, no API key) and **Civitai** (larger selection, requires API key).

#### HuggingFace (26 LoRAs, no key required)

```bash
# Download all LoRAs (one-time, ~1-2GB total)
./start_download_loras.sh

# List available LoRAs
python python/download_loras.py --list

# Download a specific LoRA
python python/download_loras.py --name moxin_ink

# Scan locally downloaded LoRAs
python python/download_loras.py --scan
```

#### Civitai (24 LoRAs, requires API key)

Get your free API key at https://civitai.com/user/account

```bash
# Download all curated Civitai LoRAs (24 style + character models)
./start_download_loras_civitai.sh

# Or with explicit API key
CIVITAI_API_KEY=your_key ./start_download_loras_civitai.sh

# Search Civitai for new LoRAs
python python/download_loras_civitai.py --search "kimono"

# Download a specific LoRA from registry
python python/download_loras_civitai.py --name detail-tweaker

# Download any Civitai model by ID
python python/download_loras_civitai.py --download 58390 62833 --output-name detail-tweaker
```

Downloaded LoRAs are saved to `python/loras/` and automatically registered in the SQLite database (`db_init.py`). The database-enhanced GUI allows selecting LoRAs with weight slider control.

### NDI Source Scanner

A GUI application to scan and list all available NDI sources on the local network.

```bash
./start_ndi_scanner.sh

# Auto-refresh every 3 seconds
./start_ndi_scanner.sh --auto-refresh 3
```

Features:
- Scan for NDI sources (10-second timeout for thorough discovery)
- Display source name, address, and URL
- View detailed source information on selection
- Auto-refresh mode

### Stream RGBD Control Panel

A GUI control panel for configuring and launching the Stream RGBD pipeline with visual parameter editing.

```bash
./start_streamrgbd_gui.sh
```

Features:
- **Visual parameter configuration** (prompt, model, render size, depth backend, NDI output, etc.)
- **Start/Stop button control**
- **Real-time prompt update** while running (green button + status label)
- **Real-time log output panel**
- All `camera_rgbd.py` arguments accessible via GUI

### Stream RGBD Database-Enhanced GUI

`stream_rgbd_gui_db.py` extends the control panel with **peewee/SQLite** database integration:

```bash
# Install peewee (included in requirements.txt)
source .venv/bin/activate
pip install peewee

# Initialize database (creates tables + seed data)
python python/db_init.py

# Launch the GUI
./start_streamrgbd_gui.sh
```

Database features:
- **Model & ModelType tables**: Manage StreamDiffusion, Depth, and LoRA model metadata
- **Categorized LoRA tables**: `StyleLoraModel`, `SubjectLoraModel`, and `QualityLoraModel` with `category`/`sub_type` fields for organized LoRA management
- **Prompt tables (3 categories)**:
  - `StylePrompt` — style descriptors (e.g. "oil painting, classical portrait")
  - `SubjectPrompt` — subject themes (e.g. "majestic dragon soaring through clouds")
  - `QualityPrompt` — quality modifiers (e.g. "masterpiece, best quality, 8k")
- **Category random buttons**: 🎲 Style / 🎲 Subject / 🎲 Quality buttons randomize each category independently
- **Combined prompt display**: Auto-joins `style + subject + quality` with comma separators; copy-to-clipboard button
- **Slider controls**: Strength / Blend / EMA with real-time value display (0.00–1.00 / 0.00–0.99)
- **Settings persistence**: Save/load GUI parameters via `AppSettings` table
- **Prompt management window**: Browse all stored prompts in a 3-tab notebook (Style/Subject/Quality)

> **Note**: The database-enhanced GUI gracefully falls back to built-in default data if the database is unavailable.

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

## Tech Stack & Dependencies

This project is built entirely on Python with the following core stack:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Runtime** | Python 3.9–3.12 | Inference backend |
| **ML Framework** | PyTorch 2.1–2.5.x (MPS) | Text encoder, model loading, device management |
| **Diffusion** | diffusers 0.21+ | Stable Diffusion pipeline components |
| **Apple Acceleration** | coremltools 7.x–8.x | CoreML model conversion & inference |
| **Vision** | OpenCV (cv2) | Camera capture, preprocessing, display window |
| **Numeric** | NumPy 1.x (< 2.0) | Tensor operations, buffer management |
| **Depth Estimation** | transformers, depth_anything_3 | Depth Anything V2/V3 backends |
| **Build** | setuptools | Python environment setup (shim for coremltools) |
| **Database** | peewee 3.17+ | SQLite ORM for models & prompt management |
| **LoRA Download** | huggingface_hub | Batch download LoRA weights from HuggingFace |

### Referenced Models & Code

| Component | Source | License |
|-----------|--------|---------|
| **SDXS-512** | [IDKiro/sdxs-512-0.9](https://huggingface.co/IDKiro/sdxs-512-0.9) | Apache-2.0 |
| **SD-Turbo** | [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) | Stability AI Community License |
| **TAESD** | [madebyollin/taesd](https://github.com/madebyollin/taesd) | MIT |
| **Depth Anything V2** | [LiheYoung/Depth-Anything-V2](https://github.com/LiheYoung/Depth-Anything-V2) | Apache-2.0 |
| **Depth Anything V3** | [depth-anything/DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL) | Apache-2.0 |
| **StreamDiffusion** (architecture reference) | [cumulo-autumn/StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) | MIT |

The 3-thread pipeline architecture (Camera / Inference / Display) and temporal-coherence techniques are derived from the original [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) project, adapted for Apple Silicon CoreML inference.

---

## Framework Migration (v2.1)

**v2.0 → v2.1**: Removed the Phoenix/Elixir web frontend. The project is now a **pure Python command-line tool** using OpenCV for real-time display.

### Why the migration?

- **macOS camera permissions** are restricted to GUI apps; Terminal.app is the simplest way to grant access
- **Reduced dependencies**: No Elixir, Phoenix, Membrane, FFmpeg, or SQLite required
- **Simpler deployment**: Single `start_streamrgbd.sh` script, no build step
- **Lower latency**: Direct Python cv2 window eliminates HTTP/WebSocket overhead

### What was removed?

| Removed | Replaced by |
|---------|-------------|
| `phx/` — Phoenix 1.7 + LiveView | Direct `python/camera_rgbd.py` |
| `StreamdiffusionMac.StreamRGBD` GenServer | `CameraApp` / `RGBDCameraApp` Python class |
| Membrane camera pipeline | `cv2.VideoCapture` |
| Erlang Port wire protocol | Direct Python threading + shared memory |
| MJPEG HTTP stream | `cv2.imshow()` window |
| PubSub status broadcast | Terminal stdout output |

The CoreML inference code (`Pipeline`, `RGBDPipeline`) and model architecture remain unchanged.

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
