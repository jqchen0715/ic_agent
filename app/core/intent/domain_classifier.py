# -*- coding: utf-8 -*-
"""IC/Verilog 领域边界分类：规则高置信命中 + LLM 结构化兜底。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol


class DomainClassifierModel(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        model_preference: str | None = None,
        **kwargs: Any,
    ) -> Any:
        ...


@dataclass(frozen=True)
class DomainClassification:
    """IC/Verilog scope 分类结果，只决定是否检索知识库。"""

    in_scope: bool
    confidence: str
    domain: str
    normalized_query: str
    reason: str
    source: str

    @property
    def should_retrieve(self) -> bool:
        return self.in_scope and self.confidence in {"high", "medium"}


class ICDomainClassifier:
    """共享领域分类器：规则只做高置信 fast-path，LLM 只做结构化分类。"""

    _IC_TERMS = (
        "verilog",
        "systemverilog",
        "rtl",
        "hdl",
        "asic",
        "fpga",
        "sdc",
        "eda",
        "vlsi",
        "sta",
        "cdc",
        "uvm",
        "dft",
        "fifo",
        "fsm",
        "axi",
        "ahb",
        "apb",
        "uart",
        "spi",
        "i2c",
        "setup",
        "hold",
        "timing",
        "slack",
        "clock",
        "reset",
        "module",
        "endmodule",
        "always",
        "assign",
        "wire",
        "reg",
        "logic",
        "posedge",
        "negedge",
        "fork",
        "join",
        "defparam",
        "create_clock",
        "$display",
        "$time",
        "乘法器",
        "加法器",
        "除法器",
        "计数器",
        "状态机",
        "流水线",
        "组合逻辑",
        "时序逻辑",
        "触发器",
        "锁存器",
        "寄存器",
        "跨时钟域",
        "亚稳态",
        "综合",
        "仿真",
        "时序",
        "约束",
        "芯片",
        "电路",
        "门级",
        "网表",
        "建立时间",
        "保持时间",
        "关键路径",
        "面积",
        "功耗",
    )
    _DOMAIN_HINTS = {
        "rtl_design": (
            "rtl",
            "verilog",
            "乘法器",
            "加法器",
            "fifo",
            "状态机",
            "流水线",
        ),
        "timing": ("sdc", "sta", "setup", "hold", "timing", "时序", "create_clock"),
        "verification": ("uvm", "testbench", "仿真", "断言", "覆盖率"),
        "eda": ("综合", "网表", "门级", "eda", "约束"),
    }

    async def classify(
        self,
        query: str,
        *,
        model_router: DomainClassifierModel | None = None,
        model_preference: str | None = None,
    ) -> DomainClassification:
        query = str(query or "").strip()
        rule_result = self.classify_by_rules(query)
        if rule_result is not None:
            return rule_result

        if model_router is None:
            return DomainClassification(
                in_scope=False,
                confidence="low",
                domain="unknown",
                normalized_query=query,
                reason="规则未命中且无 LLM 分类器可用",
                source="fallback",
            )

        return await self._classify_with_llm(query, model_router, model_preference)

    def classify_by_rules(self, query: str) -> DomainClassification | None:
        text = str(query or "").strip()
        if not text:
            return None
        lower = text.lower()
        hits = [term for term in self._IC_TERMS if term in lower or term in text]
        if not hits:
            return None

        domain = self._infer_domain(text)
        normalized_query = self._normalize_query(text, domain)
        return DomainClassification(
            in_scope=True,
            confidence="high",
            domain=domain,
            normalized_query=normalized_query,
            reason=f"规则高置信命中 IC/Verilog 术语: {', '.join(hits[:5])}",
            source="rule",
        )

    async def _classify_with_llm(
        self,
        query: str,
        model_router: DomainClassifierModel,
        model_preference: str | None,
    ) -> DomainClassification:
        system = (
            "你是 IC/Verilog 知识库的领域分类器。"
            "你只能判断问题是否应该进入 IC/Verilog 知识库检索，"
            "不能回答用户问题。"
            "只输出 JSON，不要输出解释文本。"
            "JSON 字段: in_scope(boolean), confidence(high|medium|low), "
            "domain(rtl_design|timing|verification|eda|out_of_scope|unknown), "
            "normalized_query(string), reason(string)。"
            "如果问题涉及数字 IC、RTL、Verilog/SystemVerilog、SDC/STA、EDA、"
            "数字电路模块设计或验证，则 in_scope=true。"
        )
        try:
            resp = await model_router.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
                model_preference=model_preference,
                temperature=0.0,
                max_tokens=256,
            )
            payload = _extract_json(str(getattr(resp, "content", "") or ""))
        except Exception as exc:  # noqa: BLE001
            return DomainClassification(
                in_scope=False,
                confidence="low",
                domain="unknown",
                normalized_query=query,
                reason=f"LLM 分类失败: {exc!s}",
                source="fallback",
            )

        in_scope = bool(payload.get("in_scope", False))
        confidence = str(payload.get("confidence", "low")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        domain = str(payload.get("domain", "unknown")).strip() or "unknown"
        normalized_query = str(payload.get("normalized_query", "")).strip() or query
        reason = str(payload.get("reason", "")).strip() or "LLM 结构化分类"
        if not in_scope:
            domain = "out_of_scope" if domain == "unknown" else domain

        return DomainClassification(
            in_scope=in_scope,
            confidence=confidence,
            domain=domain,
            normalized_query=normalized_query,
            reason=reason,
            source="llm",
        )

    def _infer_domain(self, query: str) -> str:
        lower = query.lower()
        for domain, terms in self._DOMAIN_HINTS.items():
            if any(term in lower or term in query for term in terms):
                return domain
        return "ic_domain"

    def _normalize_query(self, query: str, domain: str) -> str:
        if re.search(r"\b(verilog|systemverilog|rtl|ic|sdc|sta)\b", query, flags=re.I):
            return query
        domain_prefix = {
            "rtl_design": "IC/Verilog RTL",
            "timing": "IC/Verilog SDC STA",
            "verification": "IC/Verilog 验证",
            "eda": "IC/Verilog EDA",
        }.get(domain, "IC/Verilog")
        return f"{domain_prefix} {query}"


def _extract_json(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
