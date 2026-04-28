# -*- coding: utf-8 -*-
"""工具注册工厂。"""

from __future__ import annotations

from app.core.tools.registry import ToolRegistry


def build_ic_tool_registry() -> ToolRegistry:
    """构建并返回已注册 IC 三工具的 ToolRegistry。"""
    registry = ToolRegistry()
    registry.register_ic_tools()
    return registry

