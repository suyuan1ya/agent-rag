"""统一配置管理，使用 pydantic-settings 从 .env 和环境变量加载。"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    dashscope_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen-plus"
    vl_model: str = "qwen-vl-plus"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    collection_name: str = "pdf_slices"

    # Embedding
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_dim: int = 768
    embedding_device: str = "cuda"

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # RRF
    rrf_dense_weight: float = 0.6
    rrf_sparse_weight: float = 0.4

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 150

    # Agent
    max_agent_iterations: int = 10
    conversation_max_turns: int = 20
    conversation_ttl: int = 3600

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Observability
    otel_enabled: bool = False
    log_level: str = "INFO"


# 全局单例
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
