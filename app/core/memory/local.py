# -*- coding: utf-8 -*-
"""本地文件记忆后端：短期窗口 + 轻量长期召回。"""

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
from app.models.schemas import MemoryItem, Message


class LocalMemoryStore:
    """无需 Redis/Milvus 的本地 JSONL 记忆后端。

    每个会话一个 JSONL 文件，既保存短期对话消息，
    也保存可检索的长期条目。
    这个实现适合本地演示、开发和轻量部署；
    生产环境仍可替换为 Redis + 向量库。
    """

    def __init__(self, base_path: str | Path, *, window_size: int = 20) -> None:
        self.base_path = Path(base_path)
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
                    logger.warning("忽略损坏记忆行 path={} line={}", path, line_no)
                    continue
                if isinstance(item, dict):
                    entries.append(item)
        return entries

    def _append_entry_sync(self, path: Path, entry: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _rewrite_entries_sync(self, path: Path, entries: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        tmp.replace(path)

    async def get_history(self, session_id: str) -> list[Message]:
        """读取短期历史，按时间顺序返回最近 window_size 条消息。"""
        path = self._path(session_id)
        async with self._lock(session_id):
            entries = await asyncio.to_thread(self._read_entries_sync, path)

        messages: list[Message] = []
        for entry in entries:
            if entry.get("type") != "message":
                continue
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
        """追加短期消息。"""
        path = self._path(session_id)
        entry = {
            "id": str(uuid.uuid4()),
            "type": "message",
            "created_at": self._now(),
            "role": message.role.value,
            "content": message.content,
            "metadata": message.metadata,
        }
        async with self._lock(session_id):
            await asyncio.to_thread(self._append_entry_sync, path, entry)

    async def store(self, session_id: str, content: str, metadata: dict[str, Any]) -> str:
        """写入长期记忆条目，返回 memory_id。"""
        memory_id = str(uuid.uuid4())
        path = self._path(session_id)
        entry = {
            "id": memory_id,
            "type": "memory",
            "created_at": self._now(),
            "content": content.strip(),
            "metadata": {**metadata, "memory_id": memory_id},
        }
        async with self._lock(session_id):
            await asyncio.to_thread(self._append_entry_sync, path, entry)
        return memory_id

    async def recall(self, query: str, session_id: str, top_k: int = 5) -> list[MemoryItem]:
        """基于轻量词项重叠召回长期记忆。"""
        path = self._path(session_id)
        async with self._lock(session_id):
            entries = await asyncio.to_thread(self._read_entries_sync, path)

        query_terms = self._terms(query)
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, entry in enumerate(entries):
            if entry.get("type") != "memory":
                continue
            content = str(entry.get("content", "")).strip()
            if not content:
                continue
            score = self._score(query_terms, content)
            if score <= 0 and query_terms:
                continue
            scored.append((score, idx, entry))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        items: list[MemoryItem] = []
        for score, _, entry in scored[: max(1, top_k)]:
            items.append(
                MemoryItem(
                    id=str(entry.get("id", "")),
                    content=str(entry.get("content", "")),
                    score=score,
                    metadata=dict(entry.get("metadata") or {}),
                )
            )
        return items

    async def forget(self, memory_id: str) -> None:
        """删除所有会话中匹配 memory_id 的长期记忆。"""
        self.base_path.mkdir(parents=True, exist_ok=True)
        for path in self.base_path.glob("*.jsonl"):
            session_id = path.stem
            async with self._lock(session_id):
                entries = await asyncio.to_thread(self._read_entries_sync, path)
                kept = [
                    entry
                    for entry in entries
                    if not (entry.get("type") == "memory" and entry.get("id") == memory_id)
                ]
                if len(kept) != len(entries):
                    await asyncio.to_thread(self._rewrite_entries_sync, path, kept)

    def _terms(self, text: str) -> set[str]:
        text = (text or "").lower()
        terms = set(re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]", text))
        terms.update(
            text[i : i + 2]
            for i in range(max(0, len(text) - 1))
            if re.match(r"[\u4e00-\u9fff]{2}", text[i : i + 2])
        )
        return {term for term in terms if term.strip()}

    def _score(self, query_terms: set[str], content: str) -> float:
        if not query_terms:
            return 0.1
        content_terms = self._terms(content)
        if not content_terms:
            return 0.0
        overlap = query_terms & content_terms
        return len(overlap) / max(1, len(query_terms))
