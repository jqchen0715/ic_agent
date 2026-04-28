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
        tool = self.get_tool(name)
        result = await tool.execute(**(arguments or {}))

        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list, tuple)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)

    def get_tools_description(self) -> str:
        """生成所有工具的自然语言描述（用于 System Prompt）。"""
        lines: list[str] = []
        for t in self._tools.values():
            params = ", ".join(f"{p.name}: {p.type}" for p in t.parameters) or "无"
            lines.append(f"- {t.name}: {t.description}（参数: {params}）")
        return "\n".join(lines) if lines else "（当前无可用工具）"
