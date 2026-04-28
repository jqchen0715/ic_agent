# -*- coding: utf-8 -*-
"""服务端引用重写：仅保留本轮真实检索 source/page，并生成标准参考资料。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any


_REF_HEADING_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:【?\s*(?:参考资料|参考文献|引用|references?)\s*】?)\s*(?:[:：].*)?$",
    re.IGNORECASE,
)

_INLINE_SOURCE_PAGE_RE = re.compile(
    r"来源[:：]\s*(?P<source>[^\|\n，,；;]+?)\s*(?:\||,|，|;|；)\s*页(?:码)?[:：]?\s*(?P<page>[^\n，,；;]+)",
    re.IGNORECASE,
)


@dataclass
class CitationRewriteResult:
    answer: str
    references: list[dict[str, str]] = field(default_factory=list)
    removed_fake_count: int = 0
    removed_model_reference_section: bool = False


def rewrite_answer_citations(answer: str, sources: list[dict[str, Any]]) -> CitationRewriteResult:
    """
    统一引用重写策略：
    1) 移除模型自行生成的“参考资料”区块；
    2) 校验并移除正文中的伪 source/page 引用；
    3) 由服务端基于本轮检索结果重新生成“参考资料”。
    """
    raw_answer = str(answer or "").strip()
    refs = _build_reference_entries(sources)

    body, removed_section = _strip_model_reference_section(raw_answer)
    body, removed_fake = _remove_fake_inline_citations(body, refs)

    note = ""
    if removed_fake > 0:
        note = f"注：已移除 {removed_fake} 条未在本轮检索命中的引用。"

    ref_block = _render_reference_block(refs)
    parts: list[str] = []
    if body.strip():
        parts.append(body.strip())
    if note:
        parts.append(note)
    if ref_block:
        parts.append(ref_block)

    final_answer = "\n\n".join(parts).strip()
    if not final_answer:
        final_answer = "未生成有效回答。"

    return CitationRewriteResult(
        answer=final_answer,
        references=refs,
        removed_fake_count=removed_fake,
        removed_model_reference_section=removed_section,
    )


def _normalize_source(source: str) -> str:
    return os.path.basename(str(source or "").strip()).lower()


def _normalize_page(page: str) -> str:
    text = str(page or "").strip().lower()
    if not text:
        return ""
    m = re.search(r"\d+", text)
    if m:
        return m.group(0)
    return text


def _build_reference_entries(sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in sources or []:
        source = str(item.get("source", "")).strip()
        page = str(item.get("page", "")).strip() or "页码未知"
        if not source:
            continue
        key = (_normalize_source(source), _normalize_page(page))
        if key in seen:
            continue
        seen.add(key)
        refs.append({"source": source, "page": page})
    return refs


def _strip_model_reference_section(answer: str) -> tuple[str, bool]:
    if not answer:
        return "", False

    lines = answer.splitlines()
    for idx, line in enumerate(lines):
        if _REF_HEADING_RE.match(line.strip()):
            return "\n".join(lines[:idx]).rstrip(), True
    return answer.rstrip(), False


def _remove_fake_inline_citations(answer: str, refs: list[dict[str, str]]) -> tuple[str, int]:
    valid_pairs = {(_normalize_source(r["source"]), _normalize_page(r["page"])) for r in refs}
    removed_count = 0

    def _repl(match: re.Match[str]) -> str:
        nonlocal removed_count
        source = match.group("source") or ""
        page = match.group("page") or ""
        key = (_normalize_source(source), _normalize_page(page))
        if key in valid_pairs:
            return match.group(0)
        removed_count += 1
        return "【伪引用已移除】"

    rewritten = _INLINE_SOURCE_PAGE_RE.sub(_repl, answer or "")
    return rewritten, removed_count


def _render_reference_block(refs: list[dict[str, str]]) -> str:
    lines = ["参考资料（服务端生成）"]
    if not refs:
        lines.append("1. （本轮无可用引用）")
        return "\n".join(lines)
    for idx, item in enumerate(refs, 1):
        lines.append(f"{idx}. {item['source']} | {item['page']}")
    return "\n".join(lines)
