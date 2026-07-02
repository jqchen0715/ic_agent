"""应用配置：基于 pydantic-settings，支持环境变量与 .env 文件。"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="enterprise-ai-agent", description="服务名称")
    app_env: str = Field(default="development", description="运行环境")
    debug: bool = Field(default=False, description="调试模式")
    api_prefix: str = Field(default="/api/v1", description="API 前缀")
    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=8000, description="监听端口")

    openai_api_key: str = Field(default="sk-61c81f473bae450f940fc6d1b48627be", description="OpenAI API Key")
    openai_api_base: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="OpenAI 兼容 API Base",
    )
    openai_model: str = Field(default="qwen-turbo-2025-02-11", description="默认对话模型")

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/agent_db",
        description="SQLAlchemy 异步数据库 URL（推荐 postgresql+asyncpg）",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")

    milvus_host: str = Field(default="localhost", description="Milvus 主机")
    milvus_port: int = Field(default=19530, description="Milvus 端口")
    milvus_user: str = Field(default="", description="Milvus 用户名")
    milvus_password: str = Field(default="", description="Milvus 密码")
    milvus_collection_name: str = Field(
        default="agent_knowledge",
        description="默认向量集合名",
    )

    data_path: str = Field(default="data", description="本地 PDF 知识库目录")
    chroma_path: str = Field(default="chroma_db", description="Chroma 持久化目录")
    chroma_collection_name: str = Field(default="ic_expert", description="Chroma 集合名")
    embedding_model_path: str = Field(default="/Users/jiaqing/八股/ai-agent-interview-guide/ic_project/IC-Expert-agent/model/bge-m3", description="bge-m3")
    embedding_device: str = Field(default="cpu", description="cpu")
    source_mismatch_strategy: str = Field(
        default="rebuild",
        description="source 不一致策略: warn 或 rebuild",
    )
    rag_enable_reranker: bool = Field(default=True, description="是否启用 RAG CrossEncoder 重排序")
    rag_retrieval_candidate_k: int = Field(default=20, description="RAG dense 检索候选池大小")
    rag_rerank_top_k: int = Field(default=10, description="RAG 重排序后保留数量")
    rag_reranker_model: str = Field(
        default="cross-encoder/bge-reranker-v2-m3",
        description="CrossEncoder 重排序模型",
    )
    rag_reranker_device: str | None = Field(default=None, description="CrossEncoder 运行设备")

    memory_enabled: bool = Field(default=True, description="是否启用会话记忆")
    memory_backend: str = Field(default="local", description="记忆后端: local 或 milvus")
    memory_store_path: str = Field(default="data/memory", description="本地记忆持久化目录")
    memory_milvus_collection_name: str = Field(
        default="agent_memory",
        description="Milvus 长期记忆集合名",
    )
    memory_embedding_model_path: str = Field(
        default="BAAI/bge-m3",
        description="长期记忆嵌入模型路径或 HuggingFace 名称",
    )
    memory_embedding_device: str | None = Field(
        default="cpu",
        description="长期记忆嵌入设备",
    )
    memory_window_size: int = Field(default=20, description="短期记忆保留消息条数")
    memory_recall_top_k: int = Field(default=5, description="长期记忆召回条数")
    memory_remember_assistant: bool = Field(
        default=False,
        description="是否将助手回复写入长期记忆",
    )

    log_level: str = Field(default="INFO", description="日志级别")


@lru_cache
def get_settings() -> Settings:
    return Settings()
