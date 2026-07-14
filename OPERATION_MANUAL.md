# StreamRGBD-mac 操作手册

> **版本**：重构后 v2.0（LiveView + PubSub + 实时 Prompt 更新）
> **适用平台**：macOS 14+ / Apple Silicon (M1/M2/M3/M4)

---

## 目录

1. [环境准备](#1-环境准备)
2. [项目安装](#2-项目安装)
3. [模型转换（首次）](#3-模型转换首次)
4. [启动与运行](#4-启动与运行)
5. [Web 界面操作](#5-web-界面操作)
6. [API 使用](#6-api-使用)
7. [架构说明](#7-架构说明)
8. [故障排查](#8-故障排查)

---

## 1. 环境准备

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 系统 | macOS 14 (Sonoma) | macOS 15 (Sequoia) |
| 芯片 | Apple Silicon (M1+) | M3 Ultra / M4 Max |
| 内存 | 16 GB | 32 GB+ |
| 摄像头 | 内置或 USB 摄像头 | 任意 |

### 软件依赖

| 软件 | 版本 | 用途 |
|------|------|------|
| Python | 3.9 - 3.12 | 推理后端（⚠️ 不支持 3.13+） |
| Elixir | 1.15+ | Phoenix Web 服务 |
| FFmpeg | 最新 | Membrane 摄像头捕获 |
| Xcode Command Line Tools | 最新 | CoreML 编译 |

### 检查依赖

```bash
# 检查 Python 版本
python3 --version        # 应显示 3.9.x - 3.12.x

# 检查 Elixir 版本
elixir --version         # 应显示 1.15+

# 检查 FFmpeg
ffmpeg -version | head -1

# 如果没有 FFmpeg，安装：
brew install ffmpeg
```

---

## 2. 项目安装

### 2.1 Python 环境（推理后端）

```bash
cd /Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac

# 1. 运行自动安装脚本
chmod +x python/setup.sh
python/setup.sh

# 2. 激活虚拟环境
source .venv/bin/activate

# 如果安装失败，尝试 legacy 版本：
# rm -rf .venv
# python/setup.sh --legacy
```

### 2.2 Elixir 环境（Web 前端）

```bash
cd /Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac/phx

# 安装依赖 + 数据库 + 前端资源
mix setup

# 如果已有环境，只需：
# mix deps.get && mix compile
```

---

## 3. 模型转换（首次）

CoreML 模型需要预先转换，耗时约 5 分钟（一次性操作）。

```bash
# 确保 Python 环境已激活
source .venv/bin/activate

# 转换默认模型（SDXS-512，推荐）
python python/scripts/convert_models.py

# 或转换 SD-Turbo
python python/scripts/convert_models.py --model sd-turbo
```

转换后的模型保存在 `./coreml_models/` 目录。

### 可选：深度模型（Depth Anything）

```bash
# 方案 A：Depth Anything V2（PyTorch，开箱即用）
# 无需额外操作，运行时自动下载

# 方案 B：Depth Anything V3 CoreML（Apple Silicon 最快）
# 参考 README 中的 DA3 CoreML 转换步骤
```

---

## 4. 启动与运行

### 4.1 启动 Phoenix Web 服务

```bash
cd /Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac/phx
mix phx.server
```

默认监听 **http://localhost:4000**

### 4.2 启动后控制台输出

```
[info] Running StreamdiffusionMacWeb.Endpoint with Bandit 1.5.0 at 0.0.0.0:4000 (http)
[info] Access StreamdiffusionMacWeb.Endpoint at http://localhost:4000
```

---

## 5. Web 界面操作

### 5.1 打开 LiveView 控制面板

浏览器访问 **http://localhost:4000**

界面包含四个面板：

| 面板 | 功能 |
|------|------|
| **Engine** | 启动/停止推理引擎 |
| **Instance Status** | 实时显示运行状态（Running/Ready/Busy/PID/Frames） |
| **Prompt** | 输入并实时更新 AI 风格提示词 |
| **Stream Preview** | MJPEG 实时视频流预览 |

### 5.2 操作步骤

1. **点击 "Start Engine"** — 启动 Python 推理进程和 Membrane 摄像头管道
2. **等待状态显示 Ready: yes** — 表示 CoreML 模型已加载完毕
3. **Stream Preview 显示视频** — 实时看到 AI 处理后的画面
4. **在 Prompt 输入框修改提示词** — 点击 "Set" 实时推送给 Python 进程
5. **点击 "Stop Engine"** — 停止推理并释放资源

### 5.3 状态字段说明

| 字段 | 含义 |
|------|------|
| Running | 引擎是否已启动 |
| Ready | Python 模型是否加载完成 |
| Busy | 是否正在处理帧 |
| Thread ID | Python 主线程 ID |
| Process ID | Python OS 进程 ID |
| Frames Streamed | 已推送的帧数 |
| Has Frame | 是否有可用帧 |
| Error | 错误信息（如有） |

---

## 6. API 使用

### 6.1 JSON API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/stream/start` | 启动引擎 |
| POST | `/api/stream/stop` | 停止引擎 |
| POST | `/api/stream/prompt` | 修改提示词 |
| GET | `/api/stream/status` | 获取状态 |
| GET | `/api/stream/video` | MJPEG 视频流 |

### 6.2 启动引擎

```bash
curl -X POST http://127.0.0.1:4000/api/stream/start \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "oil painting style, masterpiece",
    "model": "sdxs",
    "render-size": 512,
    "output-size": 512,
    "width": 640,
    "height": 480
  }'
```

### 6.3 修改提示词

```bash
curl -X POST http://127.0.0.1:4000/api/stream/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "cyberpunk city, neon lights"}'
```

### 6.4 读取状态

```bash
curl http://127.0.0.1:4000/api/stream/status
```

返回示例：
```json
{
  "ok": true,
  "status": {
    "running": true,
    "ready": true,
    "busy": false,
    "mode": "camera",
    "prompt": "oil painting style, masterpiece",
    "instance_pid": 12345,
    "instance_tid": 123456789,
    "frame_count": 1280,
    "has_frame": true,
    "error": null
  }
}
```

### 6.5 MJPEG 视频流

```bash
# 在浏览器中直接打开
open http://127.0.0.1:4000/api/stream/video

# 或用 curl 保存帧
curl http://127.0.0.1:4000/api/stream/video > stream.mjpeg
```

### 6.6 IEx 交互式控制

```bash
cd phx && iex -S mix
```

```elixir
# 启动引擎
StreamdiffusionMac.StreamRGBD.start_engine(prompt: "oil painting style, masterpiece")

# 实时修改提示词
StreamdiffusionMac.StreamRGBD.set_prompt("watercolor painting, soft brushstrokes")

# 查看状态
{:ok, status} = StreamdiffusionMac.StreamRGBD.status()

# 停止引擎
StreamdiffusionMac.StreamRGBD.stop_engine()
```

---

## 7. 架构说明

### 7.1 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                         浏览器 (LiveView)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Start/Stop │  │  Prompt     │  │  <img src="/video">     │  │
│  │  按钮       │  │  输入框     │  │  MJPEG 实时预览         │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────────┘  │
└─────────┼────────────────┼──────────────────────────────────────┘
          │                │
          │   WebSocket    │   PubSub 广播
          │   (LiveView)   │
          ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phoenix 后端 (Elixir)                         │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  StreamRGBD (GenServer 编排器)                            │    │
│  │  ├─ start_engine/1  → 启动 InferenceWorker + CameraPipeline│  │
│  │  ├─ set_prompt/1    → 通过 Port 发送 prompt 更新           │  │
│  │  ├─ stop_engine/0   → 停止所有组件                         │  │
│  │  └─ broadcast_status/1 → PubSub 广播到所有 LiveView        │  │
│  └──────────────────────────────────────────────────────────┘    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐    │
│  │ InferenceWorker  │  │ CameraPipeline   │  │ VideoStreamer│    │
│  │ (GenServer)      │  │ (Membrane)       │  │ (GenServer)  │    │
│  │ ├─ 管理 Port    │  │ ├─ 摄像头捕获   │  │ ├─ 存储 JPEG │    │
│  │ ├─ 帧大小校验   │  │ ├─ RGB 转换     │  │ ├─ 深度帧    │    │
│  │ └─ 0xFFFFFFFF   │  │ └─ FrameSink    │  │ └─ 状态统计  │    │
│  │   prompt 协议   │  │    转发帧       │  │              │    │
│  └────────┬─────────┘  └──────────────────┘  └──────┬───────┘    │
└───────────┼──────────────────────────────────────────┼────────────┘
            │  Erlang Port (stdin/stdout)              │ MJPEG
            │  :packet, 4 长度前缀协议                   │ 视频流
            ▼                                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Python 推理后端                             │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  inference_worker.py                                      │    │
│  │  ├─ read_packet()   → 解析 0xFFFFFFFF prompt 更新        │    │
│  │  ├─ Pipeline        → CoreML img2img 推理                │    │
│  │  ├─ write_frame()   → 输出 JPEG 到 stdout                 │    │
│  │  └─ write_depth_frame() → 0xFFFFFFFE 深度帧              │    │
│  └──────────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  camera.py / camera_rgbd.py (被 inference_worker 导入)   │    │
│  │  ├─ VAE Encode    → CoreML TAESD (~5ms)                   │    │
│  │  ├─ UNet Inference → CoreML 单步去噪 (~24ms)              │    │
│  │  ├─ VAE Decode    → CoreML TAESD (~5ms)                   │    │
│  │  └─ DepthEstimator → CoreML DA3 / PyTorch DA2             │    │
│  └──────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Wire 协议详解

**Elixir → Python 帧数据：**
```
┌─────────────┬─────────────┬────────────────────────────┐
│ width: u32  │ height: u32 │ rgb_bytes: [u8; width*height*3] │
│ (little)    │ (little)    │                            │
└─────────────┴─────────────┴────────────────────────────┘
```

**Elixir → Python Prompt 更新：**
```
┌────────────────┬─────────────────┬──────────────────────────┐
│ 0xFFFFFFFF: u32│ prompt_len: u32 │ prompt: [u8; prompt_len] │
│ (sentinel)     │ (little)        │ (UTF-8)                  │
└────────────────┴─────────────────┴──────────────────────────┘
```

**Python → Elixir 普通帧：**
```
┌─────────────┬────────────────────────────┐
│ jpeg_size   │ jpeg_bytes                 │
│ u32 little  │ [u8; jpeg_size]              │
└─────────────┴────────────────────────────┘
```

**Python → Elixir 深度帧：**
```
┌────────────────┬─────────────┬────────────────────────────┐
│ 0xFFFFFFFE     │ jpeg_size   │ depth_jpeg                 │
│ u32 sentinel   │ u32 little  │ [u8; jpeg_size]              │
└────────────────┴─────────────┴────────────────────────────┘
```

**Python → Elixir Ready 信标：**
```
┌─────────────┬─────────────┬─────────────┐
│ 0: u32      │ pid: u32    │ tid: u32    │
│             │ (little)    │ (little)    │
└─────────────┴─────────────┴─────────────┘
```

### 7.3 关键进程树

```
StreamdiffusionMac.Supervisor (one_for_one)
├── StreamdiffusionMacWeb.Telemetry
├── StreamdiffusionMac.Repo
├── StreamdiffusionMac.PipelineAgent (Agent)
├── StreamdiffusionMac.VideoStreamer (GenServer)
├── StreamdiffusionMac.InferenceWorker (GenServer)
│   └── python inference_worker.py (Port 子进程)
├── StreamdiffusionMac.StreamRGBD (GenServer)
├── Phoenix.PubSub
└── StreamdiffusionMacWeb.Endpoint
    └── StreamRGBDLive (LiveView 进程，每个浏览器连接一个)
```

---

## 8. 故障排查

### 8.1 常见问题速查表

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| Python 未找到 | `.venv` 未创建或路径错误 | 运行 `python/setup.sh` 并激活 `source .venv/bin/activate` |
| 编译失败 | Elixir 版本过低 | 升级至 Elixir 1.15+ |
| 摄像头无法打开 | 权限问题或 FFmpeg 缺失 | `brew install ffmpeg`，检查系统摄像头权限 |
| 模型加载慢 | 首次运行需编译 CoreML | 等待 30-180 秒，属于正常 |
| 帧率低 | 硬件性能不足 | 降低 `--render-size` 到 384 或 320 |
| 画面闪烁 | 反馈参数不合适 | 增加 `--feedback` 到 0.2-0.3 |
| LiveView 不更新 | PubSub 未连接 | 刷新页面，检查 `mix phx.server` 日志 |
| 深度图未输出 | 深度模型未安装 | 使用 `--depth-backend=pytorch` 或转换 CoreML 模型 |

### 8.2 日志级别

```bash
# 查看详细日志
DEBUG_STREAM=1 mix phx.server

# 在 iex 中查看 Python 输出
iex -S mix
# 观察 [InferenceWorker] 开头的日志
```

### 8.3 重置环境

```bash
# 重置数据库
cd phx && mix ecto.reset

# 重新编译前端资源
cd phx && mix assets.deploy

# 清理并重新编译
cd phx && mix clean && mix compile
```

### 8.4 性能调优

```bash
# 在 iex 中调整参数启动
iex -S mix

StreamdiffusionMac.StreamRGBD.start_engine(
  prompt: "oil painting style",
  "render-size": 512,      # 可降为 384/320 提升 FPS
  "output-size": 512,
  strength: 0.5,           # 去噪强度（0.1-1.0）
  feedback: 0.1            # 时间连贯性反馈（0.0-0.3）
)
```

### 8.5 端口冲突

如果 4000 端口被占用：

```bash
# 修改端口
cd phx
PORT=4001 mix phx.server
```

---

## 附录 A：文件清单

| 文件 | 用途 |
|------|------|
| `python/camera.py` | 基础 img2img pipeline |
| `python/camera_rgbd.py` | RGBD pipeline（深度 + 风格） |
| `python/inference_worker.py` | Port 子进程（重构后支持 prompt 更新 + RGBD） |
| `python/streamdiffusion_api.py` | 独立 HTTP API（Flask） |
| `phx/lib/streamdiffusion_mac/stream_rgbd.ex` | GenServer 编排器（重构后支持 PubSub） |
| `phx/lib/streamdiffusion_mac/inference_worker.ex` | Port 管理（重构后支持 0xFFFFFFFF prompt 协议） |
| `phx/lib/streamdiffusion_mac/video_streamer.ex` | 帧存储（重构后支持深度帧） |
| `phx/lib/streamdiffusion_mac/camera_pipeline.ex` | Membrane 摄像头管道 |
| `phx/lib/streamdiffusion_mac_web/live/stream_rgbd_live.ex` | LiveView 控制面板（新建） |
| `phx/lib/streamdiffusion_mac_web/stream_rgbd_controller.ex` | JSON API + MJPEG 流 |

---

## 附录 B：Skill 引用

本项目的架构模式已封装为可复用 Skill：

**Skill 路径：** `~/.kimi/daimon/skills/stream-rgbd-elixir-python/SKILL.md`

包含内容：
- Erlang Port 双协议设计（帧 + prompt）
- Membrane → FrameSink → GenServer 流水线
- Phoenix PubSub 实时状态广播
- OTP Supervisor 进程树配置
- MJPEG 流服务实现
- 性能优化与错误处理模式

---

*手册生成时间：2025-07-14*
