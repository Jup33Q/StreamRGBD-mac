# Database & GUI 集成使用指南

本文档说明如何初始化数据库并运行带有数据库集成的 GUI 程序。

---

## 前提条件

- 项目已配置虚拟环境（`.venv`）
- `peewee` 需要安装在虚拟环境中

---

## 1. 激活虚拟环境并安装依赖

先进入项目目录：

```bash
cd /Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac
```

激活虚拟环境：

```bash
source .venv/bin/activate
```

安装 peewee（已添加到 requirements.txt）：

```bash
uv pip install peewee
```

或一次性安装全部依赖（包含 peewee）：

```bash
uv pip install -r python/requirements.txt
```

---

## 2. 初始化数据库

确保虚拟环境已激活，然后执行：

```bash
cd python
python db_init.py
```

成功输出示例：

```
============================================================
StreamDiffusion 数据库初始化
============================================================
[INFO] 数据目录已存在: /Users/.../streamdiffusion-mac/data
[INFO] 数据库连接: /Users/.../streamdiffusion-mac/data/streamdiffusion.db
[INFO] 数据库表创建完成（共 7 个表）
  ✓ model_type
  ✓ model
  ✓ prompt_category
  ✓ style_prompt
  ✓ subject_prompt
  ✓ quality_prompt
  ✓ app_settings
[INFO] ModelType 种子数据: 插入 4 条新记录，总计 4 条
[INFO] Model 种子数据: 插入 9 条新记录，总计 9 条
[INFO] PromptCategory 种子数据: 插入 3 条新记录，总计 3 条
[INFO] StylePrompt 种子数据: 插入 8 条新记录，总计 8 条
[INFO] SubjectPrompt 种子数据: 插入 5 条新记录，总计 5 条
[INFO] QualityPrompt 种子数据: 插入 7 条新记录，总计 7 条
[INFO] AppSettings 种子数据: 插入 6 条新记录，总计 6 条
[INFO] 数据库日志模式: wal
  ✓ WAL 模式已启用
============================================================
数据库初始化完成！
============================================================
```

---

## 3. 运行 GUI 程序

### 方式一：通过启动脚本（推荐）

```bash
cd /Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac
source .venv/bin/activate
./start_streamrgbd_gui.sh
```

### 方式二：直接运行 Python

```bash
cd /Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac
source .venv/bin/activate
cd python
python stream_rgbd_gui_db.py
```

---

## 4. 提示词功能说明

GUI 左侧面板提供 **按 Category 分组** 的提示词构建功能：

```
提示词构建（style + subject + quality）
├─ Style:   [oil painting...]      [🎲紫色]
├─ Subject:* [majestic dragon...]  [🎲粉红]  ← 必填（红色*）
├─ Quality: [masterpiece...]       [🎲靛蓝]
├─ 合并提示词（自动拼接）: [oil painting, majestic dragon, masterpiece] [📋]
└─ [🎲🎲🎲 全部随机]
```

### 操作说明

| 按钮 | 功能 |
|------|------|
| 🎲（紫色） | 从 StylePrompt 表随机选取一条风格提示词 |
| 🎲（粉红） | 从 SubjectPrompt 表随机选取一条主题提示词 |
| 🎲（靛蓝） | 从 QualityPrompt 表随机选取一条质量提示词 |
| 🎲🎲🎲 | 同时随机三个 category |
| 📋 | 将合并提示词复制到系统剪贴板 |

### 合并规则

- 自动按 `style, subject, quality` 顺序用逗号拼接
- 为空的 category 自动跳过
- **Subject 不能为空**，否则启动时会弹窗阻止

---

## 5. 数据库文件位置

```
streamdiffusion-mac/
├── data/
│   ├── streamdiffusion.db          # 主数据库文件
│   ├── streamdiffusion.db-shm      # WAL 共享内存文件
│   └── streamdiffusion.db-wal      # WAL 日志文件
```

---

## 6. 数据结构速查

| 表名 | 用途 | 记录数（初始化后） |
|------|------|------------------|
| `model_type` | 模型类别（lora, depth, streamdiffusion-t2i, streamdiffusion-img2img） | 4 |
| `model` | 具体模型（sdxs, sd-turbo, midas-depth 等） | 9 |
| `prompt_category` | 提示词类别引用（style, subject, quality） | 3 |
| `style_prompt` | 风格提示词（油画、水彩、赛博朋克等） | 8 |
| `subject_prompt` | 主题提示词（龙、机器人、日式庭院等） | 5 |
| `quality_prompt` | 质量提示词（masterpiece、8k、HDR 等） | 7 |
| `app_settings` | 应用设置（设备、主题、语言等） | 6 |

---

## 7. 重新初始化数据库（清空数据）

如需重置数据库：

```bash
cd /Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac
source .venv/bin/activate
rm -f data/streamdiffusion.db*
cd python
python db_init.py
```

---

## 8. 手动添加提示词

```bash
cd /Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac
source .venv/bin/activate
cd python
python -c "
from models import StylePrompt, SubjectPrompt, QualityPrompt, PromptCategory
from db_init import init_db
init_db()

style_cat = PromptCategory.get(PromptCategory.name == 'style')
StylePrompt.create(
    category=style_cat,
    prompt_text='your custom style prompt',
    tags='custom, tag',
    is_active=True
)
print('Added!')
"
```
