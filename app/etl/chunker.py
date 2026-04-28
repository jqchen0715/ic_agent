# -*- coding: utf-8 -*-
"""文档分块：固定长度 / 递归 / 按段落 / IC 定制策略。"""

from __future__ import annotations

from enum import Enum

import tiktoken
try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

from app.etl.ic_text_splitter import ICCustomTextSplitter


class ChunkStrategy(str, Enum):
    """分块策略枚举。"""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    PARAGRAPH = "paragraph"
    IC_CUSTOM = "ic_custom"


class DocumentChunker:
    """文档分块器：支持多种策略。"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = max(32, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            logger.warning(f"tiktoken 编码不可用，回退字符切分: {exc}")
            self._encoding = None
        self._ic_splitter = ICCustomTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._len,
        )

    def _len(self, text: str) -> int:
        if self._encoding is None:
            return len(text)
        return len(self._encoding.encode(text))

    def chunk(self, text: str, strategy: ChunkStrategy = ChunkStrategy.RECURSIVE) -> list[str]:
        """按策略将全文切分为块列表。"""
        text = text.strip()
        if not text:
            return []

        base_chunks: list[str]
        if strategy == ChunkStrategy.FIXED:
            base_chunks = self._chunk_fixed(text)
        elif strategy == ChunkStrategy.PARAGRAPH:
            base_chunks = self._chunk_paragraph(text)
        elif strategy == ChunkStrategy.IC_CUSTOM:
            base_chunks = self._ic_splitter.split_text(text)
        else:
            base_chunks = self._chunk_recursive(text, _depth=0)

        normalized = self._normalize_chunk_sizes(base_chunks)
        return self._apply_chunk_overlap(normalized)

    def _chunk_fixed(self, text: str) -> list[str]:
        """按固定 token 窗口切片（无 tokenizer 时回退字符切片）。"""
        if self._encoding is not None:
            token_ids = self._encoding.encode(text)
            if not token_ids:
                return []
            out: list[str] = []
            for start in range(0, len(token_ids), self.chunk_size):
                seg = token_ids[start : start + self.chunk_size]
                if not seg:
                    continue
                out.append(self._encoding.decode(seg))
            return out

        out: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + self.chunk_size)
            out.append(text[start:end])
            if end >= n:
                break
            start = end
        return out

    def _chunk_paragraph(self, text: str) -> list[str]:
        """按空行分段，再合并到目标长度。"""
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        merged: list[str] = []
        buf = ""
        for p in paras:
            candidate = (buf + "\n\n" + p).strip() if buf else p
            if self._len(candidate) <= self.chunk_size:
                buf = candidate
            else:
                if buf:
                    merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)
        return merged if merged else [text]

    def _chunk_recursive(
        self,
        text: str,
        separators: list[str] | None = None,
        *,
        _depth: int = 0,
    ) -> list[str]:
        """递归按分隔符切分，过长片段继续细分或退回固定窗口。"""
        if _depth > 24:
            return self._chunk_fixed(text)
        seps = separators or ["\n\n", "\n", "。", ". ", " "]
        if self._len(text) <= self.chunk_size:
            return [text]

        for i, sep in enumerate(seps):
            if sep not in text:
                continue
            pieces = [p for p in text.split(sep) if p.strip() or p == ""]
            merged: list[str] = []
            buf = ""
            for part in pieces:
                candidate = (buf + sep + part) if buf else part
                if self._len(candidate) <= self.chunk_size:
                    buf = candidate
                else:
                    if buf:
                        merged.extend(
                            self._chunk_recursive(
                                buf,
                                seps[i + 1 :],
                                _depth=_depth + 1,
                            )
                        )
                    buf = part
            if buf:
                merged.extend(
                    self._chunk_recursive(buf, seps[i + 1 :], _depth=_depth + 1)
                )
            return merged if merged else self._chunk_fixed(text)

        return self._chunk_fixed(text)

    def _normalize_chunk_sizes(self, chunks: list[str]) -> list[str]:
        """保证所有块不超过 chunk_size。"""
        normalized: list[str] = []
        for chunk in chunks:
            c = (chunk or "").strip()
            if not c:
                continue
            if self._len(c) <= self.chunk_size:
                normalized.append(c)
                continue
            normalized.extend(self._chunk_fixed(c))
        return normalized

    def _apply_chunk_overlap(self, chunks: list[str]) -> list[str]:
        """统一为所有策略注入 overlap，避免块间断裂。"""
        if not chunks:
            return []
        if self.chunk_overlap <= 0:
            return chunks

        out: list[str] = [chunks[0]]
        for cur in chunks[1:]:
            prefix = self._tail_by_tokens(out[-1], self.chunk_overlap)
            merged = f"{prefix}{cur}" if prefix else cur
            out.append(self._truncate_to_size(merged))
        return out

    def _tail_by_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        raw = text or ""
        if not raw:
            return ""
        if self._encoding is None:
            return raw[-max_tokens:]
        token_ids = self._encoding.encode(raw)
        if not token_ids:
            return ""
        return self._encoding.decode(token_ids[-max_tokens:])

    def _truncate_to_size(self, text: str) -> str:
        if self._len(text) <= self.chunk_size:
            return text
        if self._encoding is None:
            return text[: self.chunk_size]
        token_ids = self._encoding.encode(text)
        return self._encoding.decode(token_ids[: self.chunk_size])
