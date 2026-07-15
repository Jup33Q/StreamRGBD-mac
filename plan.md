# Plan: tkinter + peewee skills, MCP integration, database models, random prompt feature

## 目标
1. 创建 `tkinter-peewee` skill — tkinter + peewee 数据库集成技能
2. 创建 `tkinter-fast-mcp` skill — tkinter GUI 的 Fast MCP 集成技能
3. 使用 peewee 为项目创建 SQLite 数据库模型（models.py）
   - ModelType 表：包含 lora, depth, streamdiffusion-t2i, streamdiffusion-img2img 类别
   - Model 表：包含必要字段（name, path, type_id, description, is_active, created_at 等）
   - DefaultPrompt 表：存储默认提示词
4. 在 `stream_rgbd_gui.py` 中集成数据库和随机提示词功能
5. MCP 集成覆盖两个 GUI 应用

## Stage 1 — 并行创建所有技能与模型（互不依赖）

### SubAgent 1: tkinter-peewee Skill
- 路径: `~/.kimi/daimon/skills/tkinter-peewee/SKILL.md`
- 内容: tkinter 应用与 peewee ORM 集成的最佳实践、模式、工具方法

### SubAgent 2: tkinter-fast-mcp Skill
- 路径: `~/.kimi/daimon/skills/tkinter-fast-mcp/SKILL.md`
- 内容: 快速将 MCP 客户端集成到 tkinter GUI 应用的指南，包含两个示例（stream_rgbd_gui, ndi_scanner_gui）

### SubAgent 3: Database Models
- 路径: `/Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac/python/models.py`
- 内容: peewee 模型定义 + 初始化脚本 + seed 数据
- 另外创建 `/Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac/python/db_init.py` 用于初始化数据库和 seed 数据

### SubAgent 4: GUI Integration
- 修改 `/Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac/python/stream_rgbd_gui.py`
- 内容: 集成 models.py，在 Prompt 输入框旁边添加 🎲 随机提示词按钮，从 DefaultPrompt 表中随机选取
- 同时创建增强版 GUI 文件 `/Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac/python/stream_rgbd_gui_db.py` 作为完整集成版本

## Stage 2 — 验证
- 检查所有文件是否正确创建
- 确认数据库模型结构符合要求
- 确认 GUI 随机提示词功能正常工作

## 输出
- `~/.kimi/daimon/skills/tkinter-peewee/SKILL.md`
- `~/.kimi/daimon/skills/tkinter-fast-mcp/SKILL.md`
- `/Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac/python/models.py`
- `/Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac/python/db_init.py`
- `/Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac/python/stream_rgbd_gui_db.py`（增强版，集成数据库）
- 原 `/Users/jup33q/Documents/kimi/workspace/streamdiffusion-mac/python/stream_rgbd_gui.py` 添加最小改动（随机提示词功能）
