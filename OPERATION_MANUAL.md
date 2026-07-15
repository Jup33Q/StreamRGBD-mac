# StreamRGBD-mac 操作手册

> **版本**：纯 Python 命令行版（cv2 窗口 + 独立启动脚本）
> **适用平台**：macOS 14+ / Apple Silicon (M1/M2/M3/M4)

---

## 目录

1. [环境准备](#1-环境准备)
2. [项目安装](#2-项目安装)
3. [模型转换（首次）](#3-模型转换首次)
4. [启动与运行](#4-启动与运行)
5. [Python 脚本参数](#5-python-脚本参数)
6. [按键控制](#6-按键控制)
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
| Xcode Command Line Tools | 最新 | CoreML 编译 |

### 检查依赖

```bash
# 检查 Python 版本
python3 --version        # 应显示 3.9.x - 3.12.x
```

---

## 2. 项目安装

### 2.1 Python 环境

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

### 4.1 使用启动脚本（推荐）

```bash
cd /Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac

# 基础启动（默认 SDXS-512，油画风格）
./start_streamrgbd.sh

# 自定义参数
./start_streamrgbd.sh --prompt "cyberpunk city, neon lights"
./start_streamrgbd.sh --render-size 384 --blend 0.3
```

脚本会自动检测 `.venv` 并激活虚拟环境。

### 4.2 手动运行 Python 脚本

```bash
# 确保虚拟环境已激活
source .venv/bin/activate

# 基础 img2img（无深度）
python python/camera.py --prompt "oil painting style, masterpiece"

# RGBD 深度 + 风格（streamrgbd）
python python/camera_rgbd.py --prompt "oil painting style, masterpiece"
```

### 4.3 启动后

首次启动时，macOS 会弹出**摄像头权限请求**，点击**"允许"**即可。

窗口会显示实时画面：
- **左侧**：AI 处理后的输出
- **右侧**（仅 RGBD）：深度图或混合预览

终端会显示当前 FPS 和提示词。

---

## 5. Python 脚本参数

### 5.1 基础参数（camera.py / camera_rgbd.py 通用）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--prompt` | `oil painting style, masterpiece` | AI 风格提示词 |
| `--prompts` | `false` | 使用内置 10 组提示词画廊 |
| `--model` | `sdxs` | 模型：`sdxs` / `sd-turbo` |
| `--render-size` | `512` | 推理分辨率：320 / 384 / 512 |
| `--output-size` | `512` | 输出分辨率 |
| `--strength` | `0.5` | 去噪强度（0.1-1.0） |
| `--blend` | `0.0` | 摄像头混合比例（0=纯AI） |
| `--ema` | `0.4` | EMA 平滑系数 |
| `--feedback` | `0.1` | 时间连贯性反馈（0.0-0.3） |
| `--camera` | `0` | 摄像头设备 ID |
| `--coreml-dir` | `./coreml_models` | CoreML 模型目录 |

### 5.2 RGBD 专用参数（仅 camera_rgbd.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--depth-model` | `auto` | 深度模型：`auto` / `da3-small` / `da2-small` |
| `--depth-backend` | `auto` | 深度后端：`auto` / `coreml` / `pytorch` |
| `--depth-coreml-path` | `./coreml_models/da3_small.mlpackage` | CoreML 深度模型路径 |
| `--depth-preview-mode` | `mono` | 深度预览：`mono` / `alpha` / `alpha_color` / `overlay` |

### 5.3 示例命令

```bash
# 基础油画风格
python python/camera.py --prompt "oil painting style, masterpiece"

# 水彩风格，降低分辨率提升帧率
python python/camera.py --prompt "watercolor painting" --render-size 384

# 30% 摄像头混合
python python/camera.py --blend 0.3

# 更强的时间连贯性
python python/camera.py --feedback 0.25 --ema 0.7

# RGBD + 深度预览为叠加层
python python/camera_rgbd.py --depth-preview-mode overlay

# 强制使用 PyTorch DA2（无 CoreML 深度模型时）
python python/camera_rgbd.py --depth-backend pytorch --depth-model da2-small
```

---

## 6. 按键控制

运行中按以下按键控制：

| 按键 | 功能 |
|------|------|
| `q` | 退出程序 |
| `s` | 保存当前帧（`capture_<timestamp>.png`） |
| `n` / `p` | 切换下/上一个预设提示词（需 `--prompts`） |
| `+` / `-` | 增加/减少摄像头混合比例 |
| `e` / `d` | 增加/减少 EMA 平滑系数 |
| `m` | **RGBD 专用**：切换深度预览模式 |

---

## 7. 架构说明

### 7.1 数据流

```
┌─────────────────────────────────────────────────────────┐
│                      摄像头 (Camera)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  Camera      │  │  Inference   │  │  Display         │ │
│  │  Thread      │  │  Thread      │  │  Thread          │ │
│  │  (30 FPS)    │  │  (CoreML)    │  │  (blending)      │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────┘ │
│         │                 │                               │
│         └────→ latest_frame│                               │
│                         └──→ latest_ai_result             │
│                               ↑                            │
│                               └──────────┘                 │
│         ←──────→ EMA smoothing ←───────                   │
│         ←──────→ blend(camera, AI) ←────                   │
│                      ↓                                     │
│                  cv2.imshow()                              │
└─────────────────────────────────────────────────────────┘
```

### 7.2 推理流水线（每帧）

1. **预处理**：中心裁剪，缩放到 `render_size`×`render_size`，归一化
2. **VAE 编码**：图像 → 隐空间（CoreML TAESD，~5ms）
3. **加噪**：固定种子噪声，保持时间连贯性
4. **UNet 推理**：单步去噪（CoreML，~24ms for SDXS）
5. **VAE 解码**：隐空间 → 图像（CoreML TAESD，~5ms）
6. **后处理**：反归一化，缩放，显示

### 7.3 时间连贯性机制

- **固定噪声种子**：每帧使用相同的噪声模式，消除闪烁
- **Latent Feedback**：前一帧 30% 的去噪隐空间混合到当前输入
- **EMA 平滑**：显示输出做指数移动平均

### 7.4 RGBD 扩展

在基础 img2img 之后增加：
- **深度估计**：对 AI 输出运行 Depth Anything（CoreML 或 PyTorch）
- **RGBD 拼接**：RGB 3 通道 + 深度 1 通道 = 4 通道输出
- **深度预览**：右侧窗口显示深度可视化（可按 `m` 切换模式）

---

## 8. 故障排查

### 8.1 常见问题速查表

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| Python 未找到 | `.venv` 未创建 | 运行 `python/setup.sh` 并激活 `source .venv/bin/activate` |
| 摄像头无法打开 | 权限问题 | 在**系统设置 → 隐私与安全性 → 摄像头**中开启 Terminal |
| 模型加载慢 | 首次运行需编译 CoreML | 等待 30-180 秒，属于正常 |
| 帧率低 | 硬件性能不足 | 降低 `--render-size` 到 384 或 320 |
| 画面闪烁 | 反馈参数不合适 | 增加 `--feedback` 到 0.2-0.3 |
| 深度图未输出 | 深度模型未安装 | 使用 `--depth-backend=pytorch --depth-model da2-small` |
| 窗口无法显示 | 无 GUI 环境 | 在 Terminal.app 中运行，而非后台/SSH |

### 8.2 摄像头权限重置

如果之前拒绝了权限弹窗，需要手动重置：

```bash
# 方法 1：系统设置中手动开启
# 系统设置 → 隐私与安全性 → 摄像头 → 勾选 Terminal

# 方法 2：重置所有权限（需重启 Terminal）
tccutil reset Camera
```

### 8.3 性能调优

```bash
# 低分辨率 = 高帧率（M1/M2 推荐）
python python/camera_rgbd.py --render-size 384 --output-size 384

# 更小模型（SD-Turbo）
python python/camera_rgbd.py --model sd-turbo --render-size 384

# 关闭深度估计（仅 camera.py）
python python/camera.py --render-size 512
```

### 8.4 日志查看

```bash
# 基础输出
python python/camera_rgbd.py 2>&1 | tee stream.log

# 查看 CoreML 编译进度（首次运行）
python python/camera_rgbd.py --render-size 512
# 等待 "Ready!" 出现
```

---

## 附录 A：文件清单

| 文件 | 用途 |
|------|------|
| `python/camera.py` | 基础 img2img pipeline（cv2 窗口） |
| `python/camera_rgbd.py` | RGBD pipeline（深度 + 风格） |
| `python/camera_ndi.py` | NDI 输入支持 |
| `python/camera_lora.py` | LoRA 模型支持 |
| `python/streamdiffusion_api.py` | 独立 HTTP API（Flask） |
| `python/inference_worker.py` | Port 子进程（Erlang 集成用） |
| `python/scripts/convert_models.py` | CoreML 模型转换 |
| `python/scripts/convert_depth_model.py` | 深度模型 CoreML 转换 |
| `start_streamrgbd.sh` | 一键启动脚本 |
| `coreml_models/` | 转换后的 CoreML 模型目录 |

---

## 附录 B：启动脚本源码

`start_streamrgbd.sh`：

```bash
#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv/bin" ]; then
    echo "ERROR: .venv not found. Run python/setup.sh first."
    exit 1
fi

source .venv/bin/activate

python python/camera_rgbd.py \
    --prompt "oil painting style, masterpiece" \
    --render-size 512 \
    --depth-backend pytorch \
    --depth-model da2-small \
    "$@"
```

---

*手册更新：2025-07-15（移除 Phoenix 前端，改为纯 Python 命令行版）*
