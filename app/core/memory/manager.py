# -*- coding: utf-8 -*-
"""统一记忆管理：协调短期记忆与长期记忆。"""

from __future__ import annotations

from typing import Protocol

from loguru import logger

from app.models.enums import MessageRole
from app.models.schemas import MemoryContext, MemoryItem, Message


class ShortTermMemoryProtocol(Protocol):
    async def get_history(self, session_id: str) -> list[Message]:
        ...

    async def add_message(self, session_id: str, message: Message) -> None:
        ...


class LongTermMemoryProtocol(Protocol):
    async def recall(self, query: str, session_id: str, top_k: int = 5) -> list[MemoryItem]:
        ...

    async def store(self, session_id: str, content: str, metadata: dict) -> str:
        ...


class MemoryManager:
    """统一记忆管理器：协调短期记忆和长期记忆。"""

    def __init__(
        self,
        short_term: ShortTermMemoryProtocol,
        long_term: LongTermMemoryProtocol,
        *,
        recall_top_k: int = 5,
        remember_assistant: bool = False,
    ) -> None:
        """
        :param short_term: 短期记忆实现（Redis + 窗口）
        :param long_term: 长期记忆实现（向量库）
        """
        self._stm = short_term
        self._ltm = long_term
        self._recall_top_k = max(1, recall_top_k)
        self._remember_assistant = remember_assistant

    async def get_context(self, session_id: str, query: str) -> MemoryContext:
        """获取与当前查询相关的记忆上下文（短期历史 + 长期召回）。"""
        try:
            short_msgs = await self._stm.get_history(session_id)
        except Exception as e:
            logger.exception("读取短期记忆失败: {}", e)
            short_msgs = []

        try:
            long_items = await self._ltm.recall(query, session_id, top_k=self._recall_top_k)
        except Exception as e:
            logger.exception("长期记忆召回失败: {}", e)
            long_items = []

        return MemoryContext(
            session_id=session_id,
            short_term_messages=short_msgs,
            long_term_items=long_items,
        )

    async def save(self, session_id: str, message: Message) -> None:
        """将新消息写入短期记忆（滑动窗口与压缩由 ShortTermMemory 负责）。"""
        try:
            await self._stm.add_message(session_id, message)
        except Exception as e:
            logger.exception("保存短期记忆失败: {}", e)
            raise RuntimeError(f"save 失败: {e}") from e

    async def remember(self, session_id: str, message: Message) -> str | None:
        """将消息写入长期记忆。"""
        if message.role == MessageRole.ASSISTANT and not self._remember_assistant:
            return None
        if message.role not in {MessageRole.USER, MessageRole.ASSISTANT, MessageRole.SYSTEM}:
            return None
        content = message.content.strip()
        if not content:
            return None

        metadata = {
            **message.metadata,
            "role": message.role.value,
            "memory_kind": "conversation_turn",
        }
        try:
            return await self._ltm.store(session_id, content, metadata)
        except Exception as e:
            logger.exception("保存长期记忆失败: {}", e)
            return None
