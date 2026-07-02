# -*- coding: utf-8 -*-
"""短期记忆：JSONL 会话历史窗口。"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from app.models.enums import MessageRole
from app.models.schemas import Message


class ShortTermMemory:
    """本地 JSONL 短期记忆。

    每个会话一个文件，只保存对话消息，并在读取时返回最近 ``window_size`` 条。
    这个实现没有 Redis/Milvus 依赖，适合实习项目演示和本地部署。
    """

    def __init__(self, base_path: str | Path, *, window_size: int = 20) -> None:
        self.base_path = Path(base_path) / "short_term"
        self.window_size = max(1, window_size)
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, session_id: str) -> asyncio.Lock:
        key = self._safe_session_id(session_id)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _safe_session_id(self, session_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id.strip())
        return safe[:120] or "default"

    def _path(self, session_id: str) -> Path:
        return self.base_path / f"{self._safe_session_id(session_id)}.jsonl"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read_entries_sync(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("忽略损坏短期记忆行 path={} line={}", path, line_no)
                    continue
                if isinstance(item, dict):
                    entries.append(item)
        return entries

    def _append_entry_sync(self, path: Path, entry: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def get_history(self, session_id: str) -> list[Message]:
        """读取最近 ``window_size`` 条会话消息。"""
        path = self._path(session_id)
        async with self._lock(session_id):
            entries = await asyncio.to_thread(self._read_entries_sync, path)

        messages: list[Message] = []
        for entry in entries:
            try:
                role = MessageRole(str(entry.get("role", "")))
            except ValueError:
                role = MessageRole.SYSTEM
            messages.append(
                Message(
                    role=role,
                    content=str(entry.get("content", "")),
                    metadata=dict(entry.get("metadata") or {}),
                )
            )
        return messages[-self.window_size :]

    async def add_message(self, session_id: str, message: Message) -> None:
        """追加一条短期消息。"""
        path = self._path(session_id)
        entry = {
            "id": str(uuid.uuid4()),
            "created_at": self._now(),
            "role": message.role.value,
            "content": message.content,
            "metadata": message.metadata,
        }
        async with self._lock(session_id):
            await asyncio.to_thread(self._append_entry_sync, path, entry)
