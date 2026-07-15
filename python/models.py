# -*- coding: utf-8 -*-
"""
StreamDiffusion macOS 项目数据库模型定义
使用 peewee ORM 适配 SQLite 数据库，管理模型和提示词数据
"""

import os
import random
import datetime
from peewee import (
    AutoField,
    BooleanField,
    CharField,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    TextField,
    Model as PeeweeModel,
    fn,
)
from peewee import SqliteDatabase

# ─────────────────────────────────────────────────────────────
# 数据库路径：项目 data 目录下的 sqlite 文件
# ─────────────────────────────────────────────────────────────
DB_PATH = os.path.join(
    os.path.dirname(__file__),  # 当前文件所在目录 (python/)
    "..",                       # 回到项目根目录
    "data",                     # 进入 data 目录
    "streamdiffusion.db"         # 数据库文件名
)

# 使用 SqliteDatabase 支持 WAL 模式等扩展功能
# pragmas 中的 journal_mode="WAL" 启用 WAL 模式，提升并发性能
database = SqliteDatabase(
    DB_PATH,
    pragmas={
        "journal_mode": "WAL",       # 启用 WAL (Write-Ahead Logging) 模式
        "foreign_keys": 1,           # 启用外键约束
        "busy_timeout": 5000,        # 等待锁释放的超时时间（毫秒）
    },
)


# ─────────────────────────────────────────────────────────────
# 抽象基类：所有模型继承此基类，自动包含创建时间和更新时间
# ─────────────────────────────────────────────────────────────
class BaseModel(PeeweeModel):
    """
    所有 peewee 模型的抽象基类
    自动管理 created_at 和 updated_at 字段
    """
    created_at = DateTimeField(default=datetime.datetime.now)
    updated_at = DateTimeField(default=datetime.datetime.now)

    class Meta:
        # 所有继承 BaseModel 的模型都使用此数据库
        database = database
        # 标记为抽象基类，不会单独创建表
        abstract = True


# ═════════════════════════════════════════════════════════════
# StreamDiffusionModel：StreamDiffusion 模型表
# ═════════════════════════════════════════════════════════════
class StreamDiffusionModel(BaseModel):
    """
    StreamDiffusion 模型表，存储 T2I 和 Img2Img 模型信息
    """
    id = AutoField(primary_key=True)
    name = CharField(
        max_length=128,
        unique=True,
        index=True,
        verbose_name="模型标识名",
    )
    display_name = CharField(
        max_length=256,
        verbose_name="显示名称",
    )
    file_path = CharField(
        max_length=512,
        verbose_name="模型文件路径",
    )
    model_kind = CharField(
        max_length=64,
        default="t2i",
        verbose_name="模型种类",
        help_text="t2i 或 img2img",
    )
    description = TextField(null=True, verbose_name="模型描述")
    # 以 JSON 字符串形式存储额外参数，如 {"steps": 20, "cfg_scale": 7.5}
    parameters = TextField(
        null=True,
        verbose_name="额外参数（JSON 字符串）",
    )
    is_default = BooleanField(
        default=False,
        verbose_name="是否为默认模型",
    )
    is_active = BooleanField(
        default=True,
        verbose_name="是否启用",
    )

    class Meta:
        table_name = "streamdiffusion_model"

    def __str__(self):
        return f"<StreamDiffusionModel {self.name} ({self.model_kind})>"


# ═════════════════════════════════════════════════════════════
# DepthModel：深度估计模型表
# ═════════════════════════════════════════════════════════════
class DepthModel(BaseModel):
    """
    深度估计模型表，存储 Depth Anything、MiDaS 等深度模型信息
    """
    id = AutoField(primary_key=True)
    name = CharField(
        max_length=128,
        unique=True,
        index=True,
        verbose_name="模型标识名",
    )
    display_name = CharField(
        max_length=256,
        verbose_name="显示名称",
    )
    file_path = CharField(
        max_length=512,
        verbose_name="模型文件路径",
    )
    description = TextField(null=True, verbose_name="模型描述")
    parameters = TextField(
        null=True,
        verbose_name="额外参数（JSON 字符串）",
    )
    is_default = BooleanField(
        default=False,
        verbose_name="是否为默认模型",
    )
    is_active = BooleanField(
        default=True,
        verbose_name="是否启用",
    )

    class Meta:
        table_name = "depth_model"

    def __str__(self):
        return f"<DepthModel {self.name}>"


# ═════════════════════════════════════════════════════════════
# LoraModel：LoRA 模型表
# ═════════════════════════════════════════════════════════════
class LoraModel(BaseModel):
    """
    LoRA 模型表，存储低秩适应模型信息
    包含权重范围字段，用于 GUI 滑块控制
    """
    id = AutoField(primary_key=True)
    name = CharField(
        max_length=128,
        unique=True,
        index=True,
        verbose_name="模型标识名",
    )
    display_name = CharField(
        max_length=256,
        verbose_name="显示名称",
    )
    file_path = CharField(
        max_length=512,
        verbose_name="模型文件路径",
    )
    repo_id = CharField(
        max_length=256,
        null=True,
        verbose_name="HuggingFace 仓库 ID",
    )
    filename = CharField(
        max_length=256,
        null=True,
        verbose_name="HuggingFace 文件名",
    )
    # Civitai 下载专用字段
    civitai_version_id = IntegerField(
        null=True,
        verbose_name="Civitai 版本 ID",
    )
    # 来源类型：huggingface 或 civitai
    source_type = CharField(
        max_length=16,
        default="huggingface",
        verbose_name="来源类型",
        help_text="huggingface 或 civitai",
    )
    description = TextField(null=True, verbose_name="模型描述")
    parameters = TextField(
        null=True,
        verbose_name="额外参数（JSON 字符串）",
    )
    # ── 权重控制字段 ──
    weight_min = FloatField(
        default=-2.0,
        verbose_name="最小权重值",
    )
    weight_max = FloatField(
        default=2.0,
        verbose_name="最大权重值",
    )
    weight_default = FloatField(
        default=0.8,
        verbose_name="默认权重值",
    )
    trigger_words = CharField(
        max_length=256,
        null=True,
        verbose_name="触发词（逗号分隔）",
    )
    # ── 分类字段 ──
    category = CharField(
        max_length=16,
        default="subject",
        verbose_name="一级分类",
        help_text="style / subject / quality",
    )
    sub_type = CharField(
        max_length=32,
        null=True,
        verbose_name="二级分类",
        help_text="style: painting/animation/photo/other; subject: architecture/character/clothing/effect; quality: detail/adjust/enhance/other",
    )
    file_size_mb = FloatField(
        null=True,
        verbose_name="文件大小（MB）",
        help_text="下载前预估或下载后实际的文件大小",
    )
    is_default = BooleanField(
        default=False,
        verbose_name="是否为默认模型",
    )
    is_active = BooleanField(
        default=True,
        verbose_name="是否启用",
    )

    class Meta:
        table_name = "lora_model"

    def __str__(self):
        return f"<LoraModel {self.name} (weight {self.weight_min}~{self.weight_max}, default={self.weight_default})>"


# ═════════════════════════════════════════════════════════════
# StyleLoraModel：风格类 LoRA 表（按一级分类拆分）
# ═════════════════════════════════════════════════════════════
class StyleLoraModel(BaseModel):
    """
    风格类 LoRA 表，存储风格类低秩适应模型
    二级分类：painting（绘画）、animation（动漫）、photo（摄影）、other（其他）
    """
    id = AutoField(primary_key=True)
    name = CharField(
        max_length=128,
        unique=True,
        index=True,
        verbose_name="模型标识名",
    )
    display_name = CharField(
        max_length=256,
        verbose_name="显示名称",
    )
    file_path = CharField(
        max_length=512,
        verbose_name="模型文件路径",
    )
    repo_id = CharField(
        max_length=256,
        null=True,
        verbose_name="HuggingFace 仓库 ID",
    )
    filename = CharField(
        max_length=256,
        null=True,
        verbose_name="HuggingFace 文件名",
    )
    civitai_version_id = IntegerField(
        null=True,
        verbose_name="Civitai 版本 ID",
    )
    source_type = CharField(
        max_length=16,
        default="huggingface",
        verbose_name="来源类型",
        help_text="huggingface 或 civitai",
    )
    description = TextField(null=True, verbose_name="模型描述")
    parameters = TextField(
        null=True,
        verbose_name="额外参数（JSON 字符串）",
    )
    weight_min = FloatField(
        default=-2.0,
        verbose_name="最小权重值",
    )
    weight_max = FloatField(
        default=2.0,
        verbose_name="最大权重值",
    )
    weight_default = FloatField(
        default=0.8,
        verbose_name="默认权重值",
    )
    trigger_words = CharField(
        max_length=256,
        null=True,
        verbose_name="触发词（逗号分隔）",
    )
    # ── 风格二级分类 ──
    style_subtype = CharField(
        max_length=32,
        default="other",
        verbose_name="风格二级分类",
        help_text="painting / animation / photo / other",
    )
    file_size_mb = FloatField(
        null=True,
        verbose_name="文件大小（MB）",
    )
    is_default = BooleanField(
        default=False,
        verbose_name="是否为默认模型",
    )
    is_active = BooleanField(
        default=True,
        verbose_name="是否启用",
    )

    class Meta:
        table_name = "style_lora"

    def __str__(self):
        return f"<StyleLoraModel {self.name} (subtype={self.style_subtype}, weight {self.weight_min}~{self.weight_max})>"


# ═════════════════════════════════════════════════════════════
# SubjectLoraModel：主体类 LoRA 表（按一级分类拆分）
# ═════════════════════════════════════════════════════════════
class SubjectLoraModel(BaseModel):
    """
    主体类 LoRA 表，存储主体类低秩适应模型
    二级分类：architecture（建筑）、character（角色）、clothing（服装）、effect（特效）
    """
    id = AutoField(primary_key=True)
    name = CharField(
        max_length=128,
        unique=True,
        index=True,
        verbose_name="模型标识名",
    )
    display_name = CharField(
        max_length=256,
        verbose_name="显示名称",
    )
    file_path = CharField(
        max_length=512,
        verbose_name="模型文件路径",
    )
    repo_id = CharField(
        max_length=256,
        null=True,
        verbose_name="HuggingFace 仓库 ID",
    )
    filename = CharField(
        max_length=256,
        null=True,
        verbose_name="HuggingFace 文件名",
    )
    civitai_version_id = IntegerField(
        null=True,
        verbose_name="Civitai 版本 ID",
    )
    source_type = CharField(
        max_length=16,
        default="huggingface",
        verbose_name="来源类型",
        help_text="huggingface 或 civitai",
    )
    description = TextField(null=True, verbose_name="模型描述")
    parameters = TextField(
        null=True,
        verbose_name="额外参数（JSON 字符串）",
    )
    weight_min = FloatField(
        default=-2.0,
        verbose_name="最小权重值",
    )
    weight_max = FloatField(
        default=2.0,
        verbose_name="最大权重值",
    )
    weight_default = FloatField(
        default=0.8,
        verbose_name="默认权重值",
    )
    trigger_words = CharField(
        max_length=256,
        null=True,
        verbose_name="触发词（逗号分隔）",
    )
    # ── 主体二级分类 ──
    subject_subtype = CharField(
        max_length=32,
        default="other",
        verbose_name="主体二级分类",
        help_text="architecture / character / clothing / effect / other",
    )
    file_size_mb = FloatField(
        null=True,
        verbose_name="文件大小（MB）",
    )
    is_default = BooleanField(
        default=False,
        verbose_name="是否为默认模型",
    )
    is_active = BooleanField(
        default=True,
        verbose_name="是否启用",
    )

    class Meta:
        table_name = "subject_lora"

    def __str__(self):
        return f"<SubjectLoraModel {self.name} (subtype={self.subject_subtype}, weight {self.weight_min}~{self.weight_max})>"


# ═════════════════════════════════════════════════════════════
# QualityLoraModel：质量类 LoRA 表（按一级分类拆分）
# ═════════════════════════════════════════════════════════════
class QualityLoraModel(BaseModel):
    """
    质量类 LoRA 表，存储质量类低秩适应模型
    二级分类：detail（细节增强）、adjust（调整）、enhance（增强）、other（其他）
    """
    id = AutoField(primary_key=True)
    name = CharField(
        max_length=128,
        unique=True,
        index=True,
        verbose_name="模型标识名",
    )
    display_name = CharField(
        max_length=256,
        verbose_name="显示名称",
    )
    file_path = CharField(
        max_length=512,
        verbose_name="模型文件路径",
    )
    repo_id = CharField(
        max_length=256,
        null=True,
        verbose_name="HuggingFace 仓库 ID",
    )
    filename = CharField(
        max_length=256,
        null=True,
        verbose_name="HuggingFace 文件名",
    )
    civitai_version_id = IntegerField(
        null=True,
        verbose_name="Civitai 版本 ID",
    )
    source_type = CharField(
        max_length=16,
        default="huggingface",
        verbose_name="来源类型",
        help_text="huggingface 或 civitai",
    )
    description = TextField(null=True, verbose_name="模型描述")
    parameters = TextField(
        null=True,
        verbose_name="额外参数（JSON 字符串）",
    )
    weight_min = FloatField(
        default=-2.0,
        verbose_name="最小权重值",
    )
    weight_max = FloatField(
        default=2.0,
        verbose_name="最大权重值",
    )
    weight_default = FloatField(
        default=0.8,
        verbose_name="默认权重值",
    )
    trigger_words = CharField(
        max_length=256,
        null=True,
        verbose_name="触发词（逗号分隔）",
    )
    # ── 质量二级分类 ──
    quality_subtype = CharField(
        max_length=32,
        default="other",
        verbose_name="质量二级分类",
        help_text="detail / adjust / enhance / other",
    )
    file_size_mb = FloatField(
        null=True,
        verbose_name="文件大小（MB）",
    )
    is_default = BooleanField(
        default=False,
        verbose_name="是否为默认模型",
    )
    is_active = BooleanField(
        default=True,
        verbose_name="是否启用",
    )

    class Meta:
        table_name = "quality_lora"

    def __str__(self):
        return f"<QualityLoraModel {self.name} (subtype={self.quality_subtype}, weight {self.weight_min}~{self.weight_max})>"


# ═════════════════════════════════════════════════════════════
# PromptCategory：提示词类别引用表
# ═════════════════════════════════════════════════════════════
class PromptCategory(BaseModel):
    """
    提示词类别引用表，定义不同类型的提示词分类
    如：style（风格）、subject（主题）、quality（质量）
    """
    id = AutoField(primary_key=True)
    name = CharField(
        max_length=64,
        unique=True,
        index=True,
        verbose_name="类别标识名",
    )
    display_name = CharField(
        max_length=128,
        verbose_name="显示名称",
    )
    description = TextField(null=True, verbose_name="类别描述")
    table_name = CharField(
        max_length=64,
        verbose_name="对应表名",
        help_text="如 style_prompt, subject_prompt, quality_prompt",
    )
    sort_order = IntegerField(
        default=0,
        verbose_name="排序顺序",
    )
    is_active = BooleanField(
        default=True,
        verbose_name="是否启用",
    )

    class Meta:
        table_name = "prompt_category"

    def __str__(self):
        return f"<PromptCategory {self.name}: {self.display_name}>"


# ═════════════════════════════════════════════════════════════
# StylePrompt：风格提示词表
# ═════════════════════════════════════════════════════════════
class StylePrompt(BaseModel):
    """
    风格提示词表，存储风格类预设提示词模板
    """
    id = AutoField(primary_key=True)
    category = ForeignKeyField(
        PromptCategory,
        backref="style_prompts",
        on_delete="CASCADE",
        on_update="CASCADE",
        verbose_name="提示词类别",
    )
    prompt_text = TextField(
        verbose_name="提示词文本",
    )
    tags = CharField(
        max_length=256,
        null=True,
        verbose_name="标签（逗号分隔）",
    )
    is_active = BooleanField(
        default=True,
        verbose_name="是否启用",
    )
    usage_count = IntegerField(
        default=0,
        verbose_name="使用次数",
    )

    class Meta:
        table_name = "style_prompt"

    def __str__(self):
        text = self.prompt_text[:50] + "..." if len(self.prompt_text) > 50 else self.prompt_text
        return f"<StylePrompt [{self.category.name}] {text}>"


# ═════════════════════════════════════════════════════════════
# SubjectPrompt：主题提示词表
# ═════════════════════════════════════════════════════════════
class SubjectPrompt(BaseModel):
    """
    主题提示词表，存储主题类预设提示词模板
    """
    id = AutoField(primary_key=True)
    category = ForeignKeyField(
        PromptCategory,
        backref="subject_prompts",
        on_delete="CASCADE",
        on_update="CASCADE",
        verbose_name="提示词类别",
    )
    prompt_text = TextField(
        verbose_name="提示词文本",
    )
    tags = CharField(
        max_length=256,
        null=True,
        verbose_name="标签（逗号分隔）",
    )
    is_active = BooleanField(
        default=True,
        verbose_name="是否启用",
    )
    usage_count = IntegerField(
        default=0,
        verbose_name="使用次数",
    )

    class Meta:
        table_name = "subject_prompt"

    def __str__(self):
        text = self.prompt_text[:50] + "..." if len(self.prompt_text) > 50 else self.prompt_text
        return f"<SubjectPrompt [{self.category.name}] {text}>"


# ═════════════════════════════════════════════════════════════
# QualityPrompt：质量提示词表
# ═════════════════════════════════════════════════════════════
class QualityPrompt(BaseModel):
    """
    质量提示词表，存储质量类预设提示词模板
    """
    id = AutoField(primary_key=True)
    category = ForeignKeyField(
        PromptCategory,
        backref="quality_prompts",
        on_delete="CASCADE",
        on_update="CASCADE",
        verbose_name="提示词类别",
    )
    prompt_text = TextField(
        verbose_name="提示词文本",
    )
    tags = CharField(
        max_length=256,
        null=True,
        verbose_name="标签（逗号分隔）",
    )
    is_active = BooleanField(
        default=True,
        verbose_name="是否启用",
    )
    usage_count = IntegerField(
        default=0,
        verbose_name="使用次数",
    )

    class Meta:
        table_name = "quality_prompt"

    def __str__(self):
        text = self.prompt_text[:50] + "..." if len(self.prompt_text) > 50 else self.prompt_text
        return f"<QualityPrompt [{self.category.name}] {text}>"


# ═════════════════════════════════════════════════════════════
# DefaultPrompt：默认提示词表（已废弃，保留用于兼容）
# ═════════════════════════════════════════════════════════════
class DefaultPrompt(BaseModel):
    """
    ⚠️ 已废弃（Deprecated）
    原默认提示词表，已被 PromptCategory + StylePrompt/SubjectPrompt/QualityPrompt 替代。
    保留此表定义用于向后兼容，不建议在新代码中使用。
    """
    id = AutoField(primary_key=True)
    prompt_text = TextField(
        verbose_name="提示词文本",
    )
    # 分类：如 "style"(风格), "subject"(主题), "quality"(质量)
    category = CharField(
        max_length=64,
        null=True,
        index=True,
        verbose_name="分类",
    )
    # 逗号分隔的标签，如 "oil painting, classic, portrait"
    tags = CharField(
        max_length=256,
        null=True,
        verbose_name="标签（逗号分隔）",
    )
    is_active = BooleanField(
        default=True,
        verbose_name="是否启用",
    )
    usage_count = IntegerField(
        default=0,
        verbose_name="使用次数",
    )
    # DefaultPrompt 已有 created_at，不需要 updated_at

    class Meta:
        table_name = "default_prompt"

    def __str__(self):
        # 截断显示，避免过长
        text = self.prompt_text[:50] + "..." if len(self.prompt_text) > 50 else self.prompt_text
        return f"<DefaultPrompt [DEPRECATED] [{self.category}] {text}>"


# ═════════════════════════════════════════════════════════════
# AppSettings：应用设置表
# ═════════════════════════════════════════════════════════════
class AppSettings(BaseModel):
    """
    应用设置表，以键值对形式存储应用配置
    例如：key="inference.device", value="mps"
    """
    id = AutoField(primary_key=True)
    key = CharField(
        max_length=128,
        unique=True,
        index=True,
        verbose_name="设置键",
    )
    value = TextField(
        verbose_name="设置值",
    )
    # 应用设置只保留 updated_at，不需要 created_at（继承自 BaseModel）
    # created_at 由 BaseModel 提供

    class Meta:
        table_name = "app_settings"

    # ── 便捷类方法 ──

    @classmethod
    def get_value(cls, key: str, default: str = "") -> str:
        """按 key 获取设置值，不存在时返回 default。"""
        try:
            setting = cls.get(cls.key == key)
            return setting.value
        except cls.DoesNotExist:
            return default

    @classmethod
    def set_value(cls, key: str, value: str) -> None:
        """按 key 设置值，不存在则创建，存在则更新。"""
        cls.replace(key=key, value=str(value)).execute()

    def __str__(self):
        return f"<AppSettings {self.key}={self.value[:30]}...>" if len(self.value) > 30 else f"<AppSettings {self.key}={self.value}>"


# ─────────────────────────────────────────────────────────────
# 便捷函数：提示词表映射与随机选取
# ─────────────────────────────────────────────────────────────

# 类别名称到表类的映射字典
_PROMPT_TABLE_MAP = {
    "style": StylePrompt,
    "subject": SubjectPrompt,
    "quality": QualityPrompt,
}


def get_prompt_table(category_name: str):
    """
    根据类别名称返回对应的提示词表类。

    Args:
        category_name: 类别标识名，如 "style"、"subject"、"quality"

    Returns:
        对应的 peewee Model 类（StylePrompt / SubjectPrompt / QualityPrompt）

    Raises:
        ValueError: 如果 category_name 不是有效的类别名
    """
    category_name = category_name.lower().strip()
    table_class = _PROMPT_TABLE_MAP.get(category_name)
    if table_class is None:
        raise ValueError(
            f"未知的提示词类别: {category_name!r}. "
            f"支持的类别: {', '.join(_PROMPT_TABLE_MAP.keys())}"
        )
    return table_class


def get_random_prompt(category_name: str) -> str | None:
    """
    从对应类别的提示词表中随机取一条 is_active=True 的提示词，
    并自动更新其 usage_count。

    Args:
        category_name: 类别标识名，如 "style"、"subject"、"quality"

    Returns:
        随机选中的提示词文本；如果该类别没有可用提示词则返回 None
    """
    table_class = get_prompt_table(category_name)

    # 先获取活跃提示词的数量
    active_query = table_class.select().where(table_class.is_active == True)
    count = active_query.count()
    if count == 0:
        return None

    # 使用随机偏移量选取一条记录（数据库无关的通用做法）
    offset = random.randint(0, count - 1)
    prompt = (
        table_class
        .select()
        .where(table_class.is_active == True)
        .offset(offset)
        .limit(1)
        .first()
    )
    if prompt is None:
        return None

    # 原子递增使用次数
    prompt.usage_count += 1
    prompt.save(only=[table_class.usage_count])

    return prompt.prompt_text


# ─────────────────────────────────────────────────────────────
# 便捷函数：LoRA 分类表映射与查询
# ─────────────────────────────────────────────────────────────

# 一级分类名称到表类的映射字典
_LORA_CATEGORY_TABLE_MAP = {
    "style": StyleLoraModel,
    "subject": SubjectLoraModel,
    "quality": QualityLoraModel,
}


def get_lora_table(category_name: str):
    """
    根据一级分类名称返回对应的 LoRA 表类。

    Args:
        category_name: 一级分类标识名，如 "style"、"subject"、"quality"

    Returns:
        对应的 peewee Model 类（StyleLoraModel / SubjectLoraModel / QualityLoraModel）

    Raises:
        ValueError: 如果 category_name 不是有效的分类名
    """
    category_name = category_name.lower().strip()
    table_class = _LORA_CATEGORY_TABLE_MAP.get(category_name)
    if table_class is None:
        raise ValueError(
            f"未知的 LoRA 一级分类: {category_name!r}. "
            f"支持的分类: {', '.join(_LORA_CATEGORY_TABLE_MAP.keys())}"
        )
    return table_class


def get_lora_by_category(category_name: str, subtype: str = None, active_only: bool = True):
    """
    根据一级分类（及可选的二级分类）获取 LoRA 列表。

    Args:
        category_name: 一级分类，如 "style"、"subject"、"quality"
        subtype: 可选的二级分类过滤，如 "clothing"、"architecture" 等
        active_only: 是否只返回 is_active=True 的条目

    Returns:
        peewee ModelSelect 查询对象
    """
    table_class = get_lora_table(category_name)
    query = table_class.select()
    if active_only:
        query = query.where(table_class.is_active == True)
    if subtype:
        # 根据分类使用正确的字段名过滤
        subtype_field = f"{category_name}_subtype"
        query = query.where(getattr(table_class, subtype_field) == subtype)
    return query


def get_all_loras(active_only: bool = True):
    """
    从所有 3 个分类表中获取全部 LoRA 列表，按统一格式返回。

    Args:
        active_only: 是否只返回 is_active=True 的条目

    Returns:
        list[dict]: 每个 LoRA 包含统一的字段（name, display_name, category, subtype, 等）
    """
    results = []
    for category_name, table_class in _LORA_CATEGORY_TABLE_MAP.items():
        query = table_class.select()
        if active_only:
            query = query.where(table_class.is_active == True)
        for lora in query:
            # 获取二级分类字段名
            subtype_field = f"{category_name}_subtype"
            subtype = getattr(lora, subtype_field, "other")
            results.append({
                "id": lora.id,
                "name": lora.name,
                "display_name": lora.display_name,
                "category": category_name,
                "sub_type": subtype,
                "file_path": lora.file_path,
                "repo_id": lora.repo_id,
                "filename": lora.filename,
                "civitai_version_id": lora.civitai_version_id,
                "source_type": lora.source_type,
                "description": lora.description,
                "weight_min": lora.weight_min,
                "weight_max": lora.weight_max,
                "weight_default": lora.weight_default,
                "trigger_words": lora.trigger_words,
                "file_size_mb": lora.file_size_mb,
                "is_default": lora.is_default,
                "is_active": lora.is_active,
            })
    return results


# ─────────────────────────────────────────────────────────────
# 便捷函数：获取数据库连接和关闭连接
# ─────────────────────────────────────────────────────────────
def connect_db():
    """打开数据库连接（通常在应用启动时调用）"""
    if database.is_closed():
        database.connect()


def close_db():
    """关闭数据库连接（通常在应用退出时调用）"""
    if not database.is_closed():
        database.close()


def get_db():
    """获取数据库实例（供其他模块使用）"""
    return database


# 便捷别名：兼容从 models 导入 db 的代码
db = database

# ─────────────────────────────────────────────────────────────
# 向后兼容：保留旧表名引用（标记为废弃）
# ─────────────────────────────────────────────────────────────
# 注意：ModelType 和 Model 已被拆分为 StreamDiffusionModel / DepthModel / LoraModel
# 旧表定义已删除，下方不再提供兼容别名
