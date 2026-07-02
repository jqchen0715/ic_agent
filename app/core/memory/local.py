# -*- coding: utf-8 -*-
"""本地 JSONL 记忆兼容入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.memory.long_term import LongTermMemory
from app.core.memory.short_term import ShortTermMemory
from app.models.schemas import MemoryItem, Message


class LocalMemoryStore:
    """兼容旧导入的本地 JSONL 记忆包装器。

    新代码应直接使用 ``ShortTermMemory`` 和 ``LongTermMemory``；
    这个类只负责把旧的双重接口委托给两层清晰的实现。
    """

    def __init__(self, base_path: str | Path, *, window_size: int = 20) -> None:
        self.short_term = ShortTermMemory(base_path, window_size=window_size)
        self.long_term = LongTermMemory(base_path)

    async def get_history(self, session_id: str) -> list[Message]:
        return await self.short_term.get_history(session_id)

    async def add_message(self, session_id: str, message: Message) -> None:
        await self.short_term.add_message(session_id, message)

    async def store(self, session_id: str, content: str, metadata: dict[str, Any]) -> str:
        return await self.long_term.store(session_id, content, metadata)

    async def recall(self, query: str, session_id: str, top_k: int = 5) -> list[MemoryItem]:
        return await self.long_term.recall(query, session_id, top_k=top_k)

    async def forget(self, memory_id: str) -> None:
        await self.long_term.forget(memory_id)
