# -*- coding: utf-8 -*-
"""长期记忆：JSONL 关键词召回，以及可选 Milvus 向量实现。"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from app.models.schemas import MemoryItem


class LongTermMemory:
    """本地 JSONL 长期记忆。

    每个会话一个文件，写入可复用的长期记忆条目；召回时按关键词/中文 bigram
    重叠打分，取同一会话下最相关的 top_k 条。
    """

    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path) / "long_term"
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
                    logger.warning("忽略损坏长期记忆行 path={} line={}", path, line_no)
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

    async def store(self, session_id: str, content: str, metadata: dict[str, Any]) -> str:
        """写入一条长期记忆，返回 memory_id。"""
        memory_id = str(uuid.uuid4())
        path = self._path(session_id)
        entry = {
            "id": memory_id,
            "created_at": self._now(),
            "content": content.strip(),
            "metadata": {**metadata, "memory_id": memory_id},
        }
        async with self._lock(session_id):
            await asyncio.to_thread(self._append_entry_sync, path, entry)
        return memory_id

    async def recall(self, query: str, session_id: str, top_k: int = 5) -> list[MemoryItem]:
        """按简单关键词重叠召回同一会话下的长期记忆。"""
        path = self._path(session_id)
        async with self._lock(session_id):
            entries = await asyncio.to_thread(self._read_entries_sync, path)

        query_terms = self._terms(query)
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, entry in enumerate(entries):
            content = str(entry.get("content", "")).strip()
            if not content:
                continue
            score = self._score(query_terms, content)
            if score <= 0 and query_terms:
                continue
            scored.append((score, idx, entry))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [
            MemoryItem(
                id=str(entry.get("id", "")),
                content=str(entry.get("content", "")),
                score=score,
                metadata=dict(entry.get("metadata") or {}),
            )
            for score, _, entry in scored[: max(1, top_k)]
        ]

    async def forget(self, memory_id: str) -> None:
        """删除所有会话中匹配 memory_id 的长期记忆。"""
        self.base_path.mkdir(parents=True, exist_ok=True)
        for path in self.base_path.glob("*.jsonl"):
            session_id = path.stem
            async with self._lock(session_id):
                entries = await asyncio.to_thread(self._read_entries_sync, path)
                kept = [entry for entry in entries if entry.get("id") != memory_id]
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


@runtime_checkable
class LTMEmbedProtocol(Protocol):
    """嵌入模型接口。"""

    def embed_query(self, text: str) -> list[float]:
        ...


@runtime_checkable
class LTMCollectionProtocol(Protocol):
    """Milvus Collection 最小接口。"""

    def insert(self, data: Any, **kwargs: Any) -> Any:
        ...

    def search(
        self,
        data: list[list[float]],
        anns_field: str,
        param: dict[str, Any],
        limit: int,
        expr: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        ...

    def delete(self, expr: str, **kwargs: Any) -> Any:
        ...

    def flush(self, **kwargs: Any) -> Any:
        ...


class MilvusLongTermMemory:
    """可选增强：基于 Milvus 向量数据库的长期记忆。"""

    vector_field: str = "embedding"
    content_field: str = "content"
    session_field: str = "session_id"
    pk_field: str = "pk"
    meta_field: str = "meta"

    metric_param: dict[str, Any] = {"metric_type": "L2", "params": {"nprobe": 16}}

    def __init__(self, milvus_collection: Any, embedding_model: Any) -> None:
        self._coll = milvus_collection
        self._embed = embedding_model

    def _ensure_embed(self) -> None:
        if not isinstance(self._embed, LTMEmbedProtocol):
            raise TypeError("embedding_model 需实现 embed_query")

    def _ensure_coll(self) -> None:
        if not isinstance(self._coll, LTMCollectionProtocol):
            raise TypeError("milvus_collection 需支持 insert/search/delete")

    async def store(self, session_id: str, content: str, metadata: dict[str, Any]) -> str:
        """写入一条长期记忆，返回 memory_id。"""
        self._ensure_embed()
        self._ensure_coll()

        memory_id = str(uuid.uuid4())
        meta = dict(metadata)
        meta["memory_id"] = memory_id

        def _sync() -> None:
            vec = self._embed.embed_query(content)
            row = {
                self.pk_field: memory_id,
                self.vector_field: vec,
                self.content_field: content,
                self.session_field: session_id,
                self.meta_field: json.dumps(meta, ensure_ascii=False),
            }
            self._coll.insert([row])
            try:
                self._coll.flush()
            except Exception as fe:
                logger.warning("flush 失败（可忽略）: {}", fe)

        try:
            await asyncio.to_thread(_sync)
        except Exception as e:
            logger.exception("Milvus 长期记忆写入失败: {}", e)
            raise RuntimeError(f"store 失败: {e}") from e

        return memory_id

    async def recall(self, query: str, session_id: str, top_k: int = 5) -> list[MemoryItem]:
        """按语义在指定会话内召回记忆。"""
        self._ensure_embed()
        self._ensure_coll()

        def _sync() -> list[MemoryItem]:
            vec = self._embed.embed_query(query)
            if hasattr(self._coll, "load"):
                self._coll.load()
            sid = session_id.replace("\\", "\\\\").replace('"', '\\"')
            expr = f'{self.session_field} == "{sid}"'
            out = self._coll.search(
                data=[vec],
                anns_field=self.vector_field,
                param=self.metric_param,
                limit=top_k,
                expr=expr,
                output_fields=[self.pk_field, self.content_field, self.meta_field],
            )
            items: list[MemoryItem] = []
            hits = out[0] if out else []
            for hit in hits:
                entity = getattr(hit, "entity", {}) or {}
                if hasattr(hit, "entity") and not isinstance(entity, dict):
                    try:
                        entity = hit.entity.to_dict()  # type: ignore[assignment]
                    except Exception:
                        entity = {}
                rid = str(entity.get(self.pk_field) or getattr(hit, "id", ""))
                text = str(entity.get(self.content_field) or "")
                meta_raw = entity.get(self.meta_field) or "{}"
                try:
                    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
                except json.JSONDecodeError:
                    meta = {}
                score = float(getattr(hit, "distance", 0.0) or 0.0)
                items.append(MemoryItem(id=rid, content=text, score=score, metadata=meta))
            return items

        try:
            return await asyncio.to_thread(_sync)
        except Exception as e:
            logger.exception("Milvus 长期记忆召回失败: {}", e)
            raise RuntimeError(f"recall 失败: {e}") from e

    async def forget(self, memory_id: str) -> None:
        """按主键删除一条记忆。"""
        self._ensure_coll()

        def _sync() -> None:
            mid = memory_id.replace("'", "\\'")
            expr = f'{self.pk_field} == "{mid}"'
            self._coll.delete(expr)
            try:
                self._coll.flush()
            except Exception as fe:
                logger.warning("flush 失败（可忽略）: {}", fe)

        try:
            await asyncio.to_thread(_sync)
        except Exception as e:
            logger.exception("Milvus 长期记忆删除失败: {}", e)
            raise RuntimeError(f"forget 失败: {e}") from e
