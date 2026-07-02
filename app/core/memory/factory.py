# -*- coding: utf-8 -*-
"""记忆系统工厂。"""

from __future__ import annotations

from functools import lru_cache

from loguru import logger

from app.config import get_settings
from app.core.memory.local import LocalMemoryStore
from app.core.memory.manager import MemoryManager
from app.core.memory.milvus import build_milvus_long_term_memory


@lru_cache
def get_memory_manager() -> MemoryManager | None:
    """构造默认记忆管理器。

    当前默认使用本地 JSONL 后端，
    确保开发环境无需 Redis/Milvus 也能记住会话。
    """
    settings = get_settings()
    if not settings.memory_enabled:
        return None

    store = LocalMemoryStore(
        settings.memory_store_path,
        window_size=settings.memory_window_size,
    )
    long_term = store
    backend = (settings.memory_backend or "local").lower()
    if backend == "milvus":
        try:
            long_term = build_milvus_long_term_memory(
                host=settings.milvus_host,
                port=settings.milvus_port,
                collection_name=settings.memory_milvus_collection_name,
                embedding_model=settings.memory_embedding_model_path,
                embedding_device=settings.memory_embedding_device,
                user=settings.milvus_user,
                password=settings.milvus_password,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Milvus 记忆初始化失败，退回本地 JSONL 记忆: {}", exc)

    return MemoryManager(
        short_term=store,
        long_term=long_term,
        recall_top_k=settings.memory_recall_top_k,
        remember_assistant=settings.memory_remember_assistant,
    )
