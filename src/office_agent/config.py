"""集中式配置模块。

设计目标（对应 12-Factor App 的配置原则）：
1. 非敏感默认值放 config/settings.yaml，随代码进入版本库；
2. 敏感值 / 环境差异通过环境变量覆盖，例如
   OFFICE_AGENT_LLM__API_KEY 覆盖 yaml 中的 llm.api_key
   （双下划线 __ 表示嵌套层级）；
3. 应用启动时一次性完成类型校验（Fail Fast）：
   配错（如 top_k 写成字符串）在启动瞬间报错，而不是跑到检索节点才崩。

配置来源优先级（高 -> 低）：
    代码内显式传参 > 环境变量/.env > config/settings.yaml > 字段默认值
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# 项目根目录：本文件位于 <root>/src/office_agent/config.py，向上两级即根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


# ----------------------------- 各层配置模型 -----------------------------


class AppSettings(BaseModel):
    """应用级配置。"""

    name: str = "office-agent"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class LLMSettings(BaseModel):
    """生成式 LLM 配置（OpenAI 兼容协议，供应商可切换）。"""

    provider: Literal["openai_compatible"] = "openai_compatible"
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    # SecretStr：打印/写日志时自动脱敏，防止密钥泄漏到日志文件
    api_key: SecretStr = SecretStr("")
    timeout_seconds: int = Field(30, gt=0, description="单次请求超时秒数")
    max_retries: int = Field(2, ge=0, description="请求失败后的最大重试次数")
    temperature: float = Field(0.2, ge=0.0, le=2.0)


class EmbeddingSettings(BaseModel):
    """Embedding 模型配置。"""

    provider: Literal["local_bge"] = "local_bge"
    model_name: str = "BAAI/bge-m3"
    # 本地权重路径（ModelScope/HF 预下载）：非空时优先于 model_name，
    # 支持完全离线部署；相对路径相对项目根目录解析
    local_path: str = ""
    device: Literal["cpu", "cuda"] = "cpu"


class VectorStoreSettings(BaseModel):
    """向量库配置（端口-适配器架构中，换库只改这里 + 新增适配器）。"""

    provider: Literal["chroma"] = "chroma"
    persist_dir: str = "data/chroma"
    collection_name: str = "office_docs"


class IndexingSettings(BaseModel):
    """离线索引流水线参数。"""

    chunk_size: int = Field(500, gt=0, description="单切片最大字符数")
    chunk_overlap: int = Field(50, ge=0, description="相邻切片重叠字符数")
    supported_extensions: list[str] = [".md", ".txt"]
    embedding_batch_size: int = Field(8, ge=1)

    @model_validator(mode="after")
    def _check_overlap(self) -> "IndexingSettings":
        # 重叠必须小于切片大小，否则窗口切分会原地踏步/死循环
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return self


class RAGSettings(BaseModel):
    """RAG 检索相关参数。"""

    top_k: int = Field(4, ge=1, le=20)
    score_threshold: float = Field(0.5, ge=0.0, le=1.0)
    rerank_enabled: bool = False


class MeetingSettings(BaseModel):
    """会议纪要场景参数。"""

    extraction_max_retries: int = Field(2, ge=0)
    sync_confidence_threshold: float = Field(0.8, ge=0.0, le=1.0)


# ----------------------------- 聚合配置对象 -----------------------------


class Settings(BaseSettings):
    """应用启动后的唯一配置入口，全项目通过 get_settings() 获取单例。"""

    model_config = SettingsConfigDict(
        env_prefix="OFFICE_AGENT_",       # 环境变量前缀
        env_nested_delimiter="__",       # 嵌套层级分隔符
        env_file=str(PROJECT_ROOT / ".env"),  # 绝对路径，避免 CWD 漂移
        env_file_encoding="utf-8",
        extra="ignore",                  # 忽略多余环境变量，避免噪音报错
        yaml_file=DEFAULT_CONFIG_PATH,
        yaml_file_encoding="utf-8",
    )

    app: AppSettings = AppSettings()
    llm: LLMSettings = LLMSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    vectorstore: VectorStoreSettings = VectorStoreSettings()
    indexing: IndexingSettings = IndexingSettings()
    rag: RAGSettings = RAGSettings()
    meeting: MeetingSettings = MeetingSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """定义配置源优先级：显式传参 > 环境变量 > .env > YAML 文件。

        四个源全部启用，优先级从高到低排列。
        """
        if not DEFAULT_CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"未找到配置文件：{DEFAULT_CONFIG_PATH}。"
                f"请确认 config/settings.yaml 存在（工作目录应为项目根目录）。"
            )
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> "Settings":
        """把相对路径统一解析为相对项目根目录的绝对路径。

        否则"从哪个目录启动程序"会影响数据文件位置，是隐蔽 bug 的常见来源。
        """
        persist_dir = Path(self.vectorstore.persist_dir)
        if not persist_dir.is_absolute():
            self.vectorstore.persist_dir = str(
                (PROJECT_ROOT / persist_dir).resolve()
            )

        local_model = Path(self.embeddings.local_path)
        if self.embeddings.local_path and not local_model.is_absolute():
            self.embeddings.local_path = str(
                (PROJECT_ROOT / local_model).resolve()
            )
        return self

    def redacted_dict(self) -> dict:
        """返回可安全打印的配置字典（密钥脱敏）。"""
        data = self.model_dump()
        secret = self.llm.api_key.get_secret_value()
        data["llm"]["api_key"] = "***已配置***" if secret else "***未配置(空)***"
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局唯一的配置单例。

    lru_cache 保证配置只加载/校验一次；测试时可用
    get_settings.cache_clear() 配合环境变量覆盖后重新加载。
    """
    return Settings()
