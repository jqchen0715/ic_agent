# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest

from app.core.memory.long_term import LongTermMemory
from app.core.memory.manager import MemoryManager
from app.core.memory.short_term import ShortTermMemory
from app.models.enums import MessageRole
from app.models.schemas import Message


@pytest.mark.asyncio
async def test_jsonl_short_term_memory_persists_recent_history(tmp_path):
    short_term = ShortTermMemory(tmp_path, window_size=2)

    await short_term.add_message(
        "s1",
        Message(role=MessageRole.USER, content="setup 时序怎么优化"),
    )
    await short_term.add_message("s1", Message(role=MessageRole.ASSISTANT, content="先看关键路径"))
    await short_term.add_message("s1", Message(role=MessageRole.USER, content="继续展开"))

    history = await short_term.get_history("s1")

    assert [m.content for m in history] == ["先看关键路径", "继续展开"]


@pytest.mark.asyncio
async def test_jsonl_long_term_memory_recalls_by_keywords(tmp_path):
    long_term = LongTermMemory(tmp_path)

    await long_term.store("s1", "用户正在学习 setup/hold 时序优化", {"role": "user"})

    recalled = await long_term.recall("setup 优化", "s1", top_k=3)

    assert recalled
    assert recalled[0].content == "用户正在学习 setup/hold 时序优化"


@pytest.mark.asyncio
async def test_memory_manager_saves_short_and_long_term(tmp_path):
    short_term = ShortTermMemory(tmp_path, window_size=5)
    long_term = LongTermMemory(tmp_path)
    manager = MemoryManager(short_term, long_term, recall_top_k=2)
    message = Message(role=MessageRole.USER, content="记住我在看 Verilog 非阻塞赋值")

    await manager.save("s2", message)
    await manager.remember("s2", message)
    context = await manager.get_context("s2", "非阻塞赋值")

    assert len(context.short_term_messages) == 1
    assert context.long_term_items
    assert context.long_term_items[0].metadata["role"] == "user"
