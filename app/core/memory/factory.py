# -*- coding: utf-8 -*-
"""记忆系统工厂。"""

from __future__ import annotations

from functools import lru_cache

from loguru import logger

from app.config import get_settings
from app.core.memory.long_term import LongTermMemory
from app.core.memory.manager import MemoryManager
from app.core.memory.short_term import ShortTermMemory


@lru_cache
def get_memory_manager() -> MemoryManager | None:
    """构造默认记忆管理器。

    默认使用 JSONL 短期记忆 + JSONL 关键词长期记忆，
    确保开发环境无需 Redis/Milvus 也能完整演示记忆闭环。
    """
    settings = get_settings()
    if not settings.memory_enabled:
        return None

    short_term = ShortTermMemory(
        settings.memory_store_path,
        window_size=settings.memory_window_size,
    )
    long_term = LongTermMemory(settings.memory_store_path)
    backend = (settings.memory_backend or "local").lower()
    if backend == "milvus":
        try:
            from app.core.memory.milvus import build_milvus_long_term_memory

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
        short_term=short_term,
        long_term=long_term,
        recall_top_k=settings.memory_recall_top_k,
        remember_assistant=settings.memory_remember_assistant,
    )
