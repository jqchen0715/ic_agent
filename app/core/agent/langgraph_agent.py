# -*- coding: utf-8 -*-
"""LangGraph 主链路：pre_tool_router -> tool_executor -> answer_generator。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from app.core.intent import DomainClassification, ICDomainClassifier

try:
    from langgraph.graph import END, START, StateGraph

    _LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - 仅在缺依赖环境触发
    END = "__end__"
    START = "__start__"
    StateGraph = None
    _LANGGRAPH_AVAILABLE = False


class _ToolRegistryLike(Protocol):
    def list_tool_names(self) -> list[str]:
        ...

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        ...

    async def invoke_with_audit(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ...


class _ModelRouterLike(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        model_preference: str | None = None,
        **kwargs: Any,
    ) -> Any:
        ...


class AgentState(TypedDict, total=False):
    user_query: str
    messages: list[dict[str, Any]]
    model_preference: str | None
    temperature: float
    max_tokens: int | None
    route_reason: str
    selected_tools: list[str]
    needs_clarification: bool
    clarification: str
    strict_miss_marker: str
    rag_query: str
    domain_classification: dict[str, Any]
    tool_outputs: list[dict[str, Any]]
    final_answer: str
    model_id: str
    usage: dict[str, Any] | None


@dataclass
class LangGraphAgentResult:
    content: str
    model_id: str
    usage: dict[str, Any] | None = None
    selected_tools: list[str] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    route_reason: str = ""
    needs_clarification: bool = False


class LangGraphICAgent:
    """
    IC 垂类 LangGraph 主链路：
    1) pre_tool_router：根据意图决定工具；
    2) tool_executor：执行工具并收集输出；
    3) answer_generator：将工具结果注入上下文，生成最终回复。
    """

    _VERILOG_KEYWORDS = ("module", "always", "assign", "verilog", "rtl")
    _TIMING_KEYWORDS = ("sdc", "时序", "clock", "setup", "hold", "false path", "false_path")
    _KNOWLEDGE_CUES = (
        "什么",
        "为何",
        "为什么",
        "怎么",
        "如何",
        "区别",
        "原理",
        "方法",
        "有哪些",
        "优化",
        "what",
        "why",
        "how",
    )
    _SHORT_QUERY_EXACT = {"乘法器", "时序"}
    STRICT_MISS_MARKER = "【严格模式未命中】"

    def __init__(self, model_router: _ModelRouterLike, tool_registry: _ToolRegistryLike) -> None:
        self._model_router = model_router
        self._tools = tool_registry
        self._domain_classifier = ICDomainClassifier()
        self._graph = self._build_graph() if _LANGGRAPH_AVAILABLE else None

    def _build_graph(self) -> Any:
        if StateGraph is None:
            return None

        graph = StateGraph(AgentState)
        graph.add_node("pre_tool_router", self._pre_tool_router)
        graph.add_node("tool_executor", self._tool_executor)
        graph.add_node("answer_generator", self._answer_generator)
        graph.add_node("clarify", self._clarify)

        graph.add_edge(START, "pre_tool_router")
        graph.add_conditional_edges(
            "pre_tool_router",
            self._route_after_pre_tool_router,
            {
                "clarify": "clarify",
                "tool_executor": "tool_executor",
                "answer_generator": "answer_generator",
            },
        )
        graph.add_edge("tool_executor", "answer_generator")
        graph.add_edge("answer_generator", END)
        graph.add_edge("clarify", END)
        return graph.compile()

    async def run(
        self,
        *,
        messages: list[dict[str, Any]],
        model_preference: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LangGraphAgentResult:
        query = self._extract_user_query(messages)
        initial: AgentState = {
            "user_query": query,
            "messages": messages,
            "model_preference": model_preference,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "selected_tools": [],
            "tool_outputs": [],
            "route_reason": "",
            "needs_clarification": False,
            "clarification": "",
            "strict_miss_marker": "",
            "rag_query": "",
            "domain_classification": {},
        }
        final_state: AgentState
        if self._graph is None:
            final_state = await self._run_without_langgraph(initial)
        else:
            final_state = await self._graph.ainvoke(initial)

        tool_events = list(final_state.get("tool_outputs") or [])
        return LangGraphAgentResult(
            content=str(final_state.get("final_answer", "")),
            model_id=str(final_state.get("model_id", "")),
            usage=final_state.get("usage"),
            selected_tools=list(final_state.get("selected_tools") or []),
            tool_outputs=tool_events,
            sources=self._extract_sources(tool_events),
            tool_events=tool_events,
            route_reason=str(final_state.get("route_reason", "")),
            needs_clarification=bool(final_state.get("needs_clarification", False)),
        )

    async def _run_without_langgraph(self, state: AgentState) -> AgentState:
        """langgraph 依赖缺失时的顺序执行降级，保持主链路行为一致。"""
        current: AgentState = dict(state)
        current.update(await self._pre_tool_router(current))

        next_node = self._route_after_pre_tool_router(current)
        if next_node == "clarify":
            current.update(await self._clarify(current))
            return current

        if next_node == "tool_executor":
            current.update(await self._tool_executor(current))

        current.update(await self._answer_generator(current))
        return current

    def _route_after_pre_tool_router(self, state: AgentState) -> str:
        if state.get("needs_clarification"):
            return "clarify"
        if state.get("selected_tools"):
            return "tool_executor"
        return "answer_generator"

    async def _pre_tool_router(self, state: AgentState) -> AgentState:
        query = str(state.get("user_query", "")).strip()
        if not query:
            return {
                "needs_clarification": True,
                "clarification": "请先描述你的问题，我再调用合适工具帮你分析。",
                "route_reason": "empty_query",
                "selected_tools": [],
            }

        scope = await self._domain_classifier.classify(
            query,
            model_router=self._model_router,
            model_preference=state.get("model_preference"),
        )

        if self._is_short_query(query) and not scope.should_retrieve:
            return {
                "needs_clarification": True,
                "clarification": (
                    "你的问题有点短。可以补充下场景吗？例如："
                    "“乘法器时序优化有哪些方法”或“SDC 里 setup/hold 约束怎么写”。"
                ),
                "route_reason": "short_query_clarify",
                "selected_tools": [],
            }

        selected: list[str] = []
        available = set(self._tools.list_tool_names())
        route_query = scope.normalized_query or query
        q_lower = f"{query}\n{route_query}".lower()

        is_verilog = any(k in q_lower for k in self._VERILOG_KEYWORDS)
        is_timing = any(k in q_lower for k in self._TIMING_KEYWORDS)
        is_knowledge = any(k in query or k in q_lower for k in self._KNOWLEDGE_CUES)
        has_verilog_code = bool(self._extract_verilog_code(query)) or bool(
            re.search(r"\b(module|endmodule|always\s*@|assign\s+\w+\s*=)", query, flags=re.I)
        )

        if not scope.should_retrieve:
            return {
                "selected_tools": [],
                "route_reason": f"out_of_scope:{scope.source}:{scope.domain}",
                "needs_clarification": False,
                "rag_query": route_query,
                "domain_classification": _classification_dict(scope),
            }

        if "ic_rag_search" in available:
            selected.append("ic_rag_search")
        if is_verilog and has_verilog_code and "verilog_code_analyzer" in available:
            selected.append("verilog_code_analyzer")
        if is_timing and "timing_constraint_suggester" in available:
            selected.append("timing_constraint_suggester")

        reason_parts: list[str] = []
        if is_knowledge:
            reason_parts.append("knowledge")
        if is_verilog:
            reason_parts.append("verilog")
        if is_timing:
            reason_parts.append("timing")
        reason_parts.append(f"scope_{scope.source}_{scope.domain}")
        route_reason = "|".join(reason_parts) if reason_parts else "default_rag"

        return {
            "selected_tools": selected,
            "route_reason": route_reason,
            "needs_clarification": False,
            "rag_query": route_query,
            "domain_classification": _classification_dict(scope),
        }

    def _is_ic_domain_query(self, query: str) -> bool:
        return self._domain_classifier.classify_by_rules(query) is not None

    async def _tool_executor(self, state: AgentState) -> AgentState:
        query = str(state.get("user_query", "")).strip()
        rag_query = str(state.get("rag_query", "")).strip() or query
        selected_tools = list(state.get("selected_tools") or [])
        outputs: list[dict[str, Any]] = []
        strict_miss_marker = ""

        for tool_name in selected_tools:
            tool_query = rag_query if tool_name == "ic_rag_search" else query
            args = self._build_tool_args(tool_name, tool_query, state.get("messages") or [])
            try:
                audit = await self._invoke_tool_with_audit(tool_name, args)
                result = str(audit.get("result", ""))
                outputs.append(
                    {
                        "tool": tool_name,
                        "arguments": audit.get("arguments", args),
                        "result": result,
                        "ok": bool(audit.get("ok", True)),
                        "summary": audit.get("summary", ""),
                        "evidence": audit.get("evidence", []),
                        "confidence": audit.get("confidence", "unknown"),
                        "review_flags": audit.get("review_flags", []),
                    }
                )
                if not strict_miss_marker:
                    marker = self._build_strict_miss_marker(tool_name, result)
                    if marker:
                        strict_miss_marker = marker
            except Exception as exc:  # noqa: BLE001
                outputs.append(
                    {
                        "tool": tool_name,
                        "arguments": args,
                        "result": f"工具执行失败: {exc!s}",
                        "ok": False,
                    }
                )

        if strict_miss_marker:
            return {"tool_outputs": outputs, "strict_miss_marker": strict_miss_marker}
        return {"tool_outputs": outputs}

    async def _invoke_tool_with_audit(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        invoke_with_audit = getattr(self._tools, "invoke_with_audit", None)
        if callable(invoke_with_audit):
            return await invoke_with_audit(tool_name, args)

        result = await self._tools.invoke(tool_name, args)
        return {
            "tool": tool_name,
            "arguments": args,
            "result": result,
            "ok": True,
            "summary": str(result)[:360],
            "evidence": [],
            "confidence": "unknown",
            "review_flags": ["legacy_tool_without_audit"],
        }

    async def _answer_generator(self, state: AgentState) -> AgentState:
        if state.get("needs_clarification"):
            return {"final_answer": str(state.get("clarification", ""))}

        query = str(state.get("user_query", "")).strip()
        messages = list(state.get("messages") or [])
        if not messages:
            messages = [{"role": "user", "content": query}]

        tool_outputs = list(state.get("tool_outputs") or [])
        if not tool_outputs:
            if str(state.get("route_reason", "")).startswith("out_of_scope"):
                return {
                    "final_answer": self._strict_refusal_template(query, "问题不属于当前 IC/Verilog 知识库范围"),
                    "model_id": "strict_refusal",
                    "usage": None,
                }
            return {
                "final_answer": self._strict_refusal_template(query, "检索工具未返回结果"),
                "model_id": "strict_refusal",
                "usage": None,
            }

        tool_context = self._render_tool_context(tool_outputs)
        strict_marker = str(state.get("strict_miss_marker", "")).strip()
        if strict_marker.startswith(self.STRICT_MISS_MARKER):
            miss_reason = strict_marker[len(self.STRICT_MISS_MARKER) :].strip() or "知识库未命中可引用片段"
            return {
                "final_answer": self._strict_refusal_template(query, miss_reason),
                "model_id": "strict_refusal",
                "usage": None,
            }

        strict_refusal = self._build_strict_refusal_if_needed(query, state, tool_outputs)
        if strict_refusal is not None:
            return {
                "final_answer": strict_refusal,
                "model_id": "strict_refusal",
                "usage": None,
            }

        system_prompt = (
            "你是 IC/Verilog 知识库问答 Agent，必须严格基于 ic_rag_search 的检索片段回答。"
            "不要使用未在工具结果中出现的通用知识补答。"
            "每个事实结论都要能在检索片段中找到直接依据；证据不足时输出严格拒答。"
            "不要在正文中写文件名、页码、chunk_id 或参考资料区块，服务端会基于真实检索结果统一重写引用。"
            "答案应简洁，优先直接回答问题，再给出必要依据。"
        )

        llm_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": (
                    f"pre_tool_router 选择工具: {state.get('selected_tools', [])}\n"
                    f"路由原因: {state.get('route_reason', '')}\n\n"
                    f"工具结果:\n{tool_context}"
                ),
            },
            *messages,
        ]

        try:
            resp = await self._model_router.chat(
                llm_messages,
                model_preference=state.get("model_preference"),
                temperature=state.get("temperature", 0.2),
                max_tokens=state.get("max_tokens"),
            )
            return {
                "final_answer": str(getattr(resp, "content", "") or ""),
                "model_id": str(getattr(resp, "model_id", "") or ""),
                "usage": getattr(resp, "usage", None),
            }
        except Exception as exc:  # noqa: BLE001
            fallback = self._fallback_answer(query, tool_outputs, exc)
            return {"final_answer": fallback, "model_id": "", "usage": None}

    async def _clarify(self, state: AgentState) -> AgentState:
        return {"final_answer": str(state.get("clarification", ""))}

    def _is_short_query(self, query: str) -> bool:
        compact = re.sub(r"\s+", "", query)
        if compact in self._SHORT_QUERY_EXACT:
            return True
        if len(compact) <= 2:
            return True
        terms = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", query)
        return len(terms) <= 1 and len(compact) <= 8

    def _extract_user_query(self, messages: list[dict[str, Any]]) -> str:
        user_messages = [
            str(item.get("content", "")).strip()
            for item in messages or []
            if str(item.get("role", "")).lower() == "user" and str(item.get("content", "")).strip()
        ]
        if not user_messages:
            return str(messages[-1].get("content", "")).strip() if messages else ""

        current = user_messages[-1]
        previous = user_messages[-2] if len(user_messages) >= 2 else ""
        if previous and self._needs_contextual_query(current):
            return f"上文用户问题: {previous}\n当前追问: {current}"
        return current

    def _needs_contextual_query(self, query: str) -> bool:
        compact = re.sub(r"\s+", "", query or "").lower()
        if not compact:
            return False
        followup_markers = (
            "继续",
            "详细",
            "展开",
            "为什么",
            "怎么做",
            "如何做",
            "它",
            "这个",
            "上述",
            "前面",
            "再讲",
            "more",
            "continue",
            "why",
            "how",
        )
        return len(compact) <= 20 and any(marker in compact for marker in followup_markers)

    def _build_tool_args(
        self,
        tool_name: str,
        query: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if tool_name == "ic_rag_search":
            return {"query": query}

        if tool_name == "verilog_code_analyzer":
            code_block = self._extract_verilog_code(query)
            if not code_block:
                code_block = self._extract_verilog_code_from_history(messages)
            return {"verilog_code": code_block or query}

        if tool_name == "timing_constraint_suggester":
            args: dict[str, Any] = {"query": query}
            module_name = self._extract_module_name(query)
            if module_name:
                args["module_name"] = module_name
            period = self._extract_clock_period(query)
            if period is not None:
                args["clock_period_ns"] = period
            return args

        return {"query": query}

    def _extract_verilog_code(self, text: str) -> str:
        match = re.search(r"```(?:verilog|sv)?\s*([\s\S]*?)```", text or "", flags=re.I)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_verilog_code_from_history(self, messages: list[dict[str, Any]]) -> str:
        for item in reversed(messages or []):
            if str(item.get("role", "")).lower() != "user":
                continue
            content = str(item.get("content", ""))
            code = self._extract_verilog_code(content)
            if code:
                return code
        return ""

    def _extract_module_name(self, text: str) -> str | None:
        match = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)", text or "", flags=re.I)
        if match:
            return match.group(1)
        return None

    def _extract_clock_period(self, text: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*ns", (text or "").lower())
        if not match:
            return None
        try:
            value = float(match.group(1))
        except ValueError:
            return None
        return value if value > 0 else None

    def _render_tool_context(self, tool_outputs: list[dict[str, Any]]) -> str:
        if not tool_outputs:
            return "无工具输出。"

        blocks: list[str] = []
        for idx, item in enumerate(tool_outputs, 1):
            tool_name = str(item.get("tool", "unknown_tool"))
            status = "ok" if item.get("ok", False) else "error"
            result = str(item.get("result", "")).strip()
            blocks.append(f"[T{idx}] {tool_name} ({status})\n{result}")
        return "\n\n".join(blocks)

    def _fallback_answer(
        self,
        query: str,
        tool_outputs: list[dict[str, Any]],
        error: Exception,
    ) -> str:
        if not tool_outputs:
            return self._strict_refusal_template(query, f"LLM 调用失败且无工具结果: {error!s}")

        sources = self._extract_sources(tool_outputs)
        if not sources:
            return self._strict_refusal_template(query, f"LLM 调用失败且检索结果不可用: {error!s}")

        snippets = []
        for item in sources[:3]:
            content = str(item.get("content", "")).strip()
            if content:
                snippets.append(content)
        if not snippets:
            return self._strict_refusal_template(query, f"LLM 调用失败且检索片段为空: {error!s}")

        return "\n".join(snippets)


    def _build_strict_miss_marker(self, tool_name: str, result: Any) -> str | None:
        if tool_name != "ic_rag_search":
            return None

        text = str(result or "").strip()
        if not text:
            return f"{self.STRICT_MISS_MARKER}检索工具未返回结果"

        if self._is_rag_hit(text):
            return None

        reason = "知识库未命中可引用片段"
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            reason = str(payload.get("reason", "")).strip() or reason

        return f"{self.STRICT_MISS_MARKER}{reason}"

    def _build_strict_refusal_if_needed(
        self,
        query: str,
        state: AgentState,
        tool_outputs: list[dict[str, Any]],
    ) -> str | None:
        selected_tools = list(state.get("selected_tools") or [])
        if "ic_rag_search" not in selected_tools:
            return None

        rag_output = next((x for x in tool_outputs if x.get("tool") == "ic_rag_search"), None)
        if rag_output is None:
            return self._strict_refusal_template(query, "检索工具未返回结果")

        rag_text = str(rag_output.get("result", "")).strip()
        if not self._is_rag_hit(rag_text):
            return self._strict_refusal_template(query, "知识库未命中可引用片段")

        weak_reason = self._weak_evidence_reason(query, rag_text)
        if weak_reason:
            return self._strict_refusal_template(query, weak_reason)

        return None

    def _weak_evidence_reason(self, query: str, rag_text: str) -> str | None:
        try:
            payload = json.loads(rag_text)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None

        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return "知识库未命中可引用片段"

        contents = [str(item.get("content", "")).strip() for item in results if isinstance(item, dict)]
        if not contents:
            return "检索结果缺少可引用正文"
        if all(len(content) < 40 for content in contents):
            return "检索片段过短，无法支撑可靠回答"

        return None

    def _core_query_terms(self, query: str) -> list[str]:
        terms = re.findall(r"[A-Za-z][A-Za-z0-9_]+|[一-鿿]{2,}", query or "")
        stopwords = {
            "在", "中", "是否", "可以", "什么", "分别", "主要", "用途", "含义", "推荐",
            "使用", "进行", "如果", "发生", "给定", "参考", "文本", "期间", "更改",
            "verilog", "hdl", "语言", "程序", "模块", "实例", "定义", "表达式", "单行", "多行",
            "语法", "建立", "模型", "元件", "单元", "执行",
        }
        out: list[str] = []
        seen: set[str] = set()
        for term in terms:
            key = term.lower()
            if key in seen or term.lower() in stopwords or term in stopwords:
                continue
            if len(term) < 2:
                continue
            seen.add(key)
            out.append(term)
        return out[:5]

    def _is_rag_hit(self, text: str) -> bool:
        content = (text or "").strip()
        if not content:
            return False

        try:
            payload = json.loads(content)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            results = payload.get("results")
            if isinstance(results, list):
                return len(results) > 0

        no_hit_markers = (
            "知识库中未找到相关信息",
            "知识库中未找到足够相关的信息",
            "未找到相关信息",
            "未找到足够相关的信息",
        )
        if any(marker in content for marker in no_hit_markers):
            return False

        if re.search(r"\[R\d+\]\s*来源:\s*.+?\|\s*页码:", content):
            return True

        if all(token in content for token in ("content", "source", "page")):
            return True

        return False

    def _strict_refusal_template(self, query: str, reason: str) -> str:
        return (
            "【严格拒答】\n"
            "当前知识库未命中可引用证据，无法基于资料直接回答。\n"
            f"原因: {reason}\n"
            f"问题: {query}\n\n"
            "请补充更具体关键词，或更新/上传相关 PDF 后重试。"
        )

    def _extract_sources(self, tool_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for event in tool_events:
            if str(event.get("tool", "")) != "ic_rag_search":
                continue

            raw = str(event.get("result", "")).strip()
            if not raw:
                continue

            from_json = self._extract_sources_from_json(raw)
            if from_json:
                sources.extend(from_json)
                continue

            from_text = self._extract_sources_from_text(raw)
            if from_text:
                sources.extend(from_text)

        # 去重（source/page/chunk_id/content）
        uniq: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in sources:
            key = (
                str(item.get("source", "")),
                str(item.get("page", "")),
                str(item.get("chunk_id", "")),
                str(item.get("content", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            uniq.append(item)
        return uniq

    def _extract_sources_from_json(self, raw: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(raw)
        except Exception:
            return []

        candidates: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            if all(k in payload for k in ("content", "source", "page")):
                candidates.append(payload)
            elif isinstance(payload.get("results"), list):
                for item in payload["results"]:
                    if isinstance(item, dict):
                        candidates.append(item)
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    candidates.append(item)

        normalized: list[dict[str, Any]] = []
        for idx, item in enumerate(candidates, 1):
            normalized_item = self._normalize_source_item(item, idx)
            if normalized_item is not None:
                normalized.append(normalized_item)
        return normalized

    def _extract_sources_from_text(self, raw: str) -> list[dict[str, Any]]:
        pattern = re.compile(
            r"\[R\d+\]\s*来源:\s*(?P<source>.+?)\s*\|\s*页码:\s*(?P<page>.+?)\n片段:\s*(?P<content>[\s\S]*?)(?=\n\n\[R\d+\]|\Z)"
        )
        matches = list(pattern.finditer(raw))
        if not matches:
            return []

        out: list[dict[str, Any]] = []
        for idx, m in enumerate(matches, 1):
            source = m.group("source").strip()
            page = m.group("page").strip() or "页码未知"
            content = m.group("content").strip()
            if not source:
                continue
            out.append(
                {
                    "content": content,
                    "source": source,
                    "page": page,
                    "score": 0.0,
                    "chunk_id": f"{source}#{page}#r{idx}",
                }
            )
        return out

    def _normalize_source_item(self, item: dict[str, Any], idx: int) -> dict[str, Any] | None:
        source = str(item.get("source", "")).strip()
        if not source:
            return None

        page = str(item.get("page", "页码未知")).strip() or "页码未知"
        content = str(item.get("content", "")).strip()
        chunk_id = str(item.get("chunk_id", "")).strip() or f"{source}#{page}#r{idx}"

        score_raw = item.get("score", 0.0)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.0

        return {
            "content": content,
            "source": source,
            "page": page,
            "score": score,
            "chunk_id": chunk_id,
        }


def _classification_dict(scope: DomainClassification) -> dict[str, Any]:
    return {
        "in_scope": scope.in_scope,
        "confidence": scope.confidence,
        "domain": scope.domain,
        "normalized_query": scope.normalized_query,
        "reason": scope.reason,
        "source": scope.source,
        "should_retrieve": scope.should_retrieve,
    }
