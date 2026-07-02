# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest

from app.core.memory.local import LocalMemoryStore
from app.core.memory.manager import MemoryManager
from app.models.enums import MessageRole
from app.models.schemas import Message


@pytest.mark.asyncio
async def test_local_memory_store_persists_history_and_recalls(tmp_path):
    store = LocalMemoryStore(tmp_path, window_size=2)

    await store.add_message(
        "s1",
        Message(role=MessageRole.USER, content="setup 时序怎么优化"),
    )
    await store.add_message("s1", Message(role=MessageRole.ASSISTANT, content="先看关键路径"))
    await store.add_message("s1", Message(role=MessageRole.USER, content="继续展开"))
    await store.store("s1", "用户正在学习 setup/hold 时序优化", {"role": "user"})

    history = await store.get_history("s1")
    recalled = await store.recall("setup 优化", "s1", top_k=3)

    assert [m.content for m in history] == ["先看关键路径", "继续展开"]
    assert recalled
    assert recalled[0].content == "用户正在学习 setup/hold 时序优化"


@pytest.mark.asyncio
async def test_memory_manager_saves_short_and_long_term(tmp_path):
    store = LocalMemoryStore(tmp_path, window_size=5)
    manager = MemoryManager(store, store, recall_top_k=2)
    message = Message(role=MessageRole.USER, content="记住我在看 Verilog 非阻塞赋值")

    await manager.save("s2", message)
    await manager.remember("s2", message)
    context = await manager.get_context("s2", "非阻塞赋值")

    assert len(context.short_term_messages) == 1
    assert context.long_term_items
    assert context.long_term_items[0].metadata["role"] == "user"
