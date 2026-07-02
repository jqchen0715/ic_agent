# -*- coding: utf-8 -*-
"""工具基类：统一 name、description、parameters 与 execute 接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolParameter(BaseModel):
    """JSON Schema 风格的单个参数描述（简化）。"""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True


class BaseTool(ABC):
    """所有 Agent 工具的抽象基类。"""

    name: str = "base_tool"
    description: str = "基础工具"
    risk_level: str = "medium"

    def __init__(self) -> None:
        self.parameters: list[ToolParameter] = []

    def schema_parameters(self) -> dict[str, Any]:
        """导出为 OpenAI tools 风格的 parameters 结构。"""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            properties[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                required.append(p.name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def validate_arguments(self, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """执行前做轻量 schema 校验，避免模型把坏参数直接送进工具。"""
        values = dict(arguments or {})
        known = {p.name: p for p in self.parameters}

        extra = sorted(set(values) - set(known))
        if extra:
            raise ValueError(f"工具 {self.name} 收到未知参数: {', '.join(extra)}")

        for param in self.parameters:
            value = values.get(param.name)
            if param.required and (value is None or (isinstance(value, str) and not value.strip())):
                raise ValueError(f"工具 {self.name} 缺少必填参数: {param.name}")
            if value is None:
                continue
            if param.type == "string" and not isinstance(value, str):
                values[param.name] = str(value)
            elif param.type == "number":
                try:
                    values[param.name] = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"工具 {self.name} 参数 {param.name} 必须是数字") from exc

        return values

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """执行工具逻辑；子类实现具体行为。"""
