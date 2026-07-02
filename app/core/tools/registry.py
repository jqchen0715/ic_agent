# -*- coding: utf-8 -*-
"""工具注册中心：集中管理可用工具并生成 Prompt 描述。"""

from __future__ import annotations

import json
from typing import Any

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

from app.core.tools.base import BaseTool


class ToolRegistry:
    """工具注册中心：管理所有可用工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册工具；同名覆盖并记录日志。"""
        if tool.name in self._tools:
            logger.warning("工具 [{}] 已存在，将被覆盖", tool.name)
        self._tools[tool.name] = tool
        logger.info("已注册工具: {}", tool.name)

    def get_tool(self, name: str) -> BaseTool:
        """按名称获取工具。"""
        if name not in self._tools:
            raise KeyError(f"未注册的工具: {name}")
        return self._tools[name]

    def get_all_tools(self) -> list[BaseTool]:
        """返回全部工具列表。"""
        return list(self._tools.values())

    def list_tool_names(self) -> list[str]:
        """返回全部工具名。"""
        return list(self._tools.keys())

    def register_ic_tools(self) -> None:
        """统一注册 IC 三工具（迁移自 IC-Expert-agent/rag_core.py）。"""
        from app.core.tools.builtin.ic_tools import (  # Lazy import to avoid heavy deps at module import.
            ICRAGSearchTool,
            TimingConstraintSuggesterTool,
            VerilogCodeAnalyzerTool,
        )

        self.register(ICRAGSearchTool())
        self.register(VerilogCodeAnalyzerTool())
        self.register(TimingConstraintSuggesterTool())

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        """统一执行工具并转为字符串结果，便于注入 LLM 上下文。"""
        record = await self.invoke_with_audit(name, arguments)
        return str(record.get("result", ""))

    async def invoke_with_audit(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """执行工具并返回可审计记录，供强自主 Agent 判定可靠性。"""
        tool = self.get_tool(name)
        safe_arguments = tool.validate_arguments(arguments or {})
        result = await tool.execute(**safe_arguments)
        return self._build_audit_record(tool, safe_arguments, result)

    def _build_audit_record(
        self,
        tool: BaseTool,
        arguments: dict[str, Any],
        result: Any,
    ) -> dict[str, Any]:
        evidence = self._extract_evidence(result)
        review_flags = self._extract_review_flags(result)
        confidence = self._extract_confidence(result, evidence, review_flags)
        summary = self._extract_summary(result)
        serialized = self._serialize_result(result)
        ok = not review_flags or not any(str(flag).startswith("tool_error") for flag in review_flags)
        return {
            "tool": tool.name,
            "arguments": arguments,
            "result": serialized,
            "raw_result": result,
            "ok": ok,
            "summary": summary,
            "evidence": evidence,
            "confidence": confidence,
            "review_flags": review_flags,
            "risk_level": getattr(tool, "risk_level", "medium"),
        }

    def _serialize_result(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list, tuple)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)

    def _extract_summary(self, result: Any, max_len: int = 360) -> str:
        if isinstance(result, dict):
            for key in ("summary", "reason", "generated_sdc"):
                value = result.get(key)
                if value:
                    text = " ".join(str(value).split())
                    return text[:max_len] + ("..." if len(text) > max_len else "")
            if isinstance(result.get("findings"), list):
                findings = [str(item.get("message", "")) for item in result["findings"] if isinstance(item, dict)]
                if findings:
                    text = "；".join(findings)
                    return text[:max_len] + ("..." if len(text) > max_len else "")
        text = " ".join(self._serialize_result(result).split())
        return text[:max_len] + ("..." if len(text) > max_len else "")

    def _extract_evidence(self, result: Any) -> list[dict[str, Any]]:
        if not isinstance(result, dict):
            return []

        explicit = result.get("evidence")
        if isinstance(explicit, list):
            return [item for item in explicit if isinstance(item, dict)]

        rag_results = result.get("results")
        if isinstance(rag_results, list):
            evidence: list[dict[str, Any]] = []
            for item in rag_results:
                if not isinstance(item, dict):
                    continue
                evidence.append(
                    {
                        "type": "retrieval",
                        "source": item.get("source", ""),
                        "page": item.get("page", "页码未知"),
                        "chunk_id": item.get("chunk_id", ""),
                        "score": item.get("score", 0.0),
                        "content": item.get("content", ""),
                    }
                )
            return evidence

        return []

    def _extract_review_flags(self, result: Any) -> list[str]:
        if not isinstance(result, dict):
            return []
        flags = result.get("review_flags")
        if isinstance(flags, list):
            return [str(item) for item in flags if str(item).strip()]
        return []

    def _extract_confidence(
        self,
        result: Any,
        evidence: list[dict[str, Any]],
        review_flags: list[str],
    ) -> str:
        if isinstance(result, dict):
            confidence = str(result.get("confidence", "")).lower()
            if confidence in {"high", "medium", "low", "unknown"}:
                return confidence
        if review_flags:
            return "low"
        if evidence:
            return "high" if len(evidence) >= 2 else "medium"
        return "unknown"

    def get_tools_description(self) -> str:
        """生成所有工具的自然语言描述（用于 System Prompt）。"""
        lines: list[str] = []
        for t in self._tools.values():
            params = ", ".join(f"{p.name}: {p.type}" for p in t.parameters) or "无"
            lines.append(f"- {t.name}: {t.description}（参数: {params}）")
        return "\n".join(lines) if lines else "（当前无可用工具）"
