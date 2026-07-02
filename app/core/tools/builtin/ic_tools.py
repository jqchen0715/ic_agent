# -*- coding: utf-8 -*-
"""IC 领域工具：知识检索、Verilog 审查、时序约束建议。"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from app.config import get_settings
from app.core.rag.retriever import ICRAGRetriever
from app.core.tools.base import BaseTool, ToolParameter
from app.models.schemas import ICRetrievalResult


def _expand_ic_query(query: str) -> str:
    """对过短/过泛查询做轻量扩展，提升 IC 主题召回。"""
    q = (query or "").strip()
    if not q:
        return q

    q_lower = q.lower()
    expansions = {
        "乘法器": "乘法器 乘法单元 booth wallace 阵列乘法器 关键路径 时序优化",
        "时序": "时序分析 setup hold 关键路径 约束 优化",
        "verilog": "verilog hdl rtl 语法 综合 仿真 代码规范",
        "fork": "fork join 并行语句 initial 仿真 不可综合",
        "defparam": "defparam 参数重定义 编译时 重新定义参数值 不可综合",
        "parameter": "parameter 参数 常量 仿真期间 更改 非法",
        "参数": "parameter 参数 常量 参数值 仿真期间 更改 非法",
        "wait": "wait 语句 表达式 为假 0 x 延迟 下一条语句",
        "udp": "udp user defined primitive ASIC 单元库 元件 建模 仿真",
        "注释": "单行注释 // 多行注释 /* */ verilog 注释语法",
        "实例": "模块实例 实例数组 范围 多个子实例",
        "综合": "可综合 不可综合 语句 synthesis",
        "仿真": "simulation 仿真 模型 testbench",
        "phil": "Phil Moorby Gateway Design Automation Verilog HDL 首创",
        "moorby": "Phil Moorby Gateway Design Automation Verilog HDL 首创",
    }

    extras: list[str] = []
    for key, extra in expansions.items():
        if key in q or key in q_lower:
            extras.append(extra)
    if not extras and any(token in q for token in ("单行", "多行")):
        extras.append("单行注释 // 多行注释 /* */ verilog 注释语法")

    if not extras:
        return q
    return " ".join([q, *extras])


def _extract_query_terms(query: str) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", query or "")
    seen = set()
    result: list[str] = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result


def _focus_keywords_by_query(query: str) -> list[str]:
    q = (query or "").lower()
    keywords: list[str] = []
    if "乘法器" in query or "multiplier" in q:
        keywords.extend(
            [
                "booth",
                "wallace",
                "dadda",
                "阵列乘法器",
                "树形乘法器",
                "关键路径",
                "流水线",
                "时序优化",
                "面积",
                "功耗",
                "dsp",
                "部分积",
            ]
        )
    if "时序" in query or "timing" in q:
        keywords.extend(["setup", "hold", "关键路径", "时钟", "约束", "pipeline"])
    if "verilog" in q:
        keywords.extend(["always", "assign", "non-blocking", "组合逻辑", "时序逻辑"])

    domain_focus = {
        "fork": ["fork", "join", "并行", "不可综合", "仿真"],
        "defparam": ["defparam", "重新定义", "参数值", "编译", "不可综合"],
        "parameter": ["parameter", "参数", "常量", "仿真期间", "非法"],
        "参数": ["parameter", "参数", "常量", "仿真期间", "非法"],
        "wait": ["wait", "表达式", "为假", "0", "x", "延迟", "下一条语句"],
        "udp": ["udp", "ASIC", "单元库", "元件", "模型", "仿真"],
        "注释": ["//", "/*", "*/", "单行注释", "多行注释"],
        "实例": ["模块实例", "范围", "多个", "子实例", "实例数组"],
        "phil": ["Phil", "Moorby", "Gateway", "首创"],
        "moorby": ["Phil", "Moorby", "Gateway", "首创"],
    }
    for trigger, values in domain_focus.items():
        if trigger in q or trigger in query:
            keywords.extend(values)

    seen = set()
    uniq: list[str] = []
    for k in keywords:
        kk = k.lower()
        if kk not in seen:
            seen.add(kk)
            uniq.append(k)
    return uniq


def _extract_relevant_snippet(text: str, query: str, max_len: int = 640) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return ""

    query_terms = [t.lower() for t in _extract_query_terms(query)]
    focus_terms = [t.lower() for t in _focus_keywords_by_query(query)]
    terms = query_terms + focus_terms
    if not terms:
        return compact[:max_len] + ("..." if len(compact) > max_len else "")

    sentences = [s.strip() for s in re.split(r"(?<=[。；;.!?])\s+|\n+", compact) if s.strip()]
    scored: list[tuple[int, int, str]] = []
    for idx, s in enumerate(sentences):
        s_norm = s.lower()
        score = sum(3 * s_norm.count(t) for t in query_terms)
        score += sum(s_norm.count(t) for t in focus_terms)
        if score > 0:
            scored.append((score, idx, s))

    if not scored:
        return compact[:max_len] + ("..." if len(compact) > max_len else "")

    scored.sort(key=lambda x: x[0], reverse=True)
    selected_idx: set[int] = set()
    for _, idx, _ in scored[:4]:
        selected_idx.add(idx)
        if idx > 0:
            selected_idx.add(idx - 1)
        if idx + 1 < len(sentences):
            selected_idx.add(idx + 1)

    selected: list[str] = []
    cur_len = 0
    for idx in sorted(selected_idx):
        s = sentences[idx]
        add_len = len(s) + (1 if selected else 0)
        if cur_len + add_len > max_len:
            continue
        selected.append(s)
        cur_len += add_len

    return " ".join(selected) if selected else compact[:max_len]


def _result_identity(item: ICRetrievalResult) -> tuple[str, str, str]:
    return (str(item.source), str(item.page), str(item.chunk_id))


def _weighted_result_score(item: ICRetrievalResult, query: str) -> float:
    text = (item.content or "").lower()
    terms = _extract_query_terms(query)
    focus_terms = _focus_keywords_by_query(query)
    score = 0.0
    for term in terms:
        t = term.lower()
        if len(t) <= 1:
            continue
        score += 6.0 * text.count(t)
        if t in text[:240]:
            score += 3.0
    for term in focus_terms:
        t = term.lower()
        if len(t) <= 1:
            continue
        score += 2.0 * text.count(t)
    score += max(0.0, float(item.score or 0.0))
    return score


def _rank_results_by_terms(results: list[ICRetrievalResult], query: str) -> list[ICRetrievalResult]:
    best_by_key: dict[tuple[str, str, str], ICRetrievalResult] = {}
    for item in results:
        key = _result_identity(item)
        prev = best_by_key.get(key)
        if prev is None or _weighted_result_score(item, query) > _weighted_result_score(prev, query):
            best_by_key[key] = item

    return sorted(
        best_by_key.values(),
        key=lambda item: (_weighted_result_score(item, query), float(item.score or 0.0)),
        reverse=True,
    )


def _guess_clock_period_ns(text: str, default: float = 5.0) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*ns", (text or "").lower())
    if m:
        try:
            value = float(m.group(1))
            if value > 0:
                return value
        except ValueError:
            return default
    return default


class ICRAGSearchTool(BaseTool):
    """IC 领域知识检索工具（迁移自 rag_core.py 的 ic_rag_search）。"""

    def __init__(self, retriever: ICRAGRetriever | None = None, top_k: int = 3) -> None:
        super().__init__()
        self.name = "ic_rag_search"
        self.description = "IC领域专业知识检索工具"
        self.risk_level = "high"
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="用户问题或检索关键词",
                required=True,
            )
        ]
        self._retriever = retriever
        self._top_k = max(1, top_k)

    def _ensure_retriever(self) -> ICRAGRetriever:
        if self._retriever is not None:
            return self._retriever

        settings = get_settings()
        self._retriever = ICRAGRetriever(
            data_dir=settings.data_path,
            chroma_path=settings.chroma_path,
            collection_name=settings.chroma_collection_name,
            embedding_model=settings.embedding_model_path,
            embedding_device=settings.embedding_device,
            mismatch_strategy=settings.source_mismatch_strategy,
            enable_reranker=settings.rag_enable_reranker,
            retrieval_candidate_k=settings.rag_retrieval_candidate_k,
            rerank_top_k=settings.rag_rerank_top_k,
            reranker_model=settings.rag_reranker_model,
            reranker_device=settings.rag_reranker_device,
        )
        return self._retriever

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            raise ValueError("参数 query 不能为空")

        expanded_query = _expand_ic_query(query)
        retriever = self._ensure_retriever()
        results = await asyncio.to_thread(retriever.retrieve, expanded_query, max(self._top_k * 3, 10))
        if not results:
            return {
                "query": query,
                "expanded_query": expanded_query,
                "results": [],
                "evidence": [],
                "confidence": "low",
                "review_flags": ["rag_no_results"],
                "summary": "知识库中未找到相关信息。",
                "reason": "知识库中未找到相关信息。",
            }

        selected = sorted(
            results,
            key=lambda item: (float(item.score or 0.0), _weighted_result_score(item, query)),
            reverse=True,
        )[: max(self._top_k * 3, 8)]

        structured: list[dict[str, Any]] = []
        seen_refs: set[tuple[str, str, str]] = set()
        max_results = min(8, max(self._top_k * 2, 5))
        for item in selected:
            ref_key = (str(item.source), str(item.page), str(item.chunk_id))
            if ref_key in seen_refs:
                continue
            seen_refs.add(ref_key)
            snippet = _extract_relevant_snippet(item.content, query, max_len=900)
            if not snippet:
                continue
            structured.append(
                {
                    "content": snippet,
                    "source": item.source,
                    "page": item.page,
                    "score": float(item.score),
                    "chunk_id": item.chunk_id,
                }
            )
            if len(structured) >= max_results:
                break

        if not structured:
            return {
                "query": query,
                "expanded_query": expanded_query,
                "results": [],
                "evidence": [],
                "confidence": "low",
                "review_flags": ["rag_weak_evidence"],
                "summary": "知识库中未找到足够相关的信息。",
                "reason": "知识库中未找到足够相关的信息，请补充更具体的问题（例如：乘法器的结构/时序优化/Verilog实现）。",
            }

        return {
            "query": query,
            "expanded_query": expanded_query,
            "results": structured,
            "evidence": [
                {
                    "type": "retrieval",
                    "source": item["source"],
                    "page": item["page"],
                    "chunk_id": item["chunk_id"],
                    "score": item["score"],
                    "content": item["content"],
                }
                for item in structured
            ],
            "confidence": "high" if len(structured) >= 2 else "medium",
            "review_flags": [],
            "summary": f"检索到 {len(structured)} 条可引用知识库片段。",
        }


class VerilogCodeAnalyzerTool(BaseTool):
    """Verilog 代码基础静态审查工具（迁移自 rag_core.py）。"""

    def __init__(self) -> None:
        super().__init__()
        self.name = "verilog_code_analyzer"
        self.description = "Verilog代码静态审查工具"
        self.risk_level = "high"
        self.parameters = [
            ToolParameter(
                name="verilog_code",
                type="string",
                description="待审查的 Verilog 代码或问题描述",
                required=True,
            )
        ]

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        verilog_code = str(kwargs.get("verilog_code", "")).strip()
        if not verilog_code:
            raise ValueError("参数 verilog_code 不能为空")

        findings: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        review_flags: list[str] = []
        code_lower = verilog_code.lower()
        lines = verilog_code.splitlines()

        def add_finding(
            severity: str,
            message: str,
            flag: str | None = None,
            line_no: int | None = None,
            snippet: str | None = None,
        ) -> None:
            finding: dict[str, Any] = {"severity": severity, "message": message}
            if line_no is not None:
                finding["line"] = line_no
            if snippet:
                finding["snippet"] = snippet.strip()
            findings.append(finding)
            if flag:
                review_flags.append(flag)
            if line_no is not None or snippet:
                evidence.append(
                    {
                        "type": "code",
                        "source": "user_input",
                        "line": line_no,
                        "content": (snippet or "").strip(),
                    }
                )

        always_blocks = _extract_always_blocks(verilog_code)

        for start_line, header, body in always_blocks:
            header_lower = header.lower()
            body_lower = body.lower()
            if ("@*" in header_lower or "@(*)" in header_lower or "always_comb" in header_lower) and (
                ("if" in body_lower and "else" not in body_lower)
                or ("case" in body_lower and "default" not in body_lower)
            ):
                add_finding(
                    "warning",
                    "组合逻辑 always 块可能缺少完整 else/default，存在 latch 推断风险。",
                    "possible_latch",
                    start_line,
                    header,
                )

            if "always @" in header_lower and "@*" not in header_lower and "@(*)" not in header_lower:
                sensitivity = _extract_sensitivity(header)
                if not any(edge in sensitivity for edge in ("posedge", "negedge")):
                    add_finding(
                        "warning",
                        "组合逻辑建议使用 always_comb 或 always @(*)，避免不完整敏感列表。",
                        "incomplete_sensitivity_list",
                        start_line,
                        header,
                    )

            has_nonblocking = "<=" in body
            has_blocking = bool(re.search(r"(?<![<>=!])=(?!=)", body))
            if has_nonblocking and has_blocking:
                add_finding(
                    "warning",
                    "同一个 always 块中混用阻塞赋值和非阻塞赋值，请确认组合/时序边界。",
                    "mixed_blocking_nonblocking",
                    start_line,
                    header,
                )

        reset_terms = ("rst", "reset")
        has_reset = any(term in code_lower for term in reset_terms)
        if has_reset:
            async_reset = any(
                any(term in _extract_sensitivity(header).lower() for term in reset_terms)
                for _, header, _ in always_blocks
            )
            if async_reset:
                add_finding("info", "检测到异步复位敏感列表，请确认复位极性与项目规范一致。")
            else:
                add_finding(
                    "info",
                    "检测到同步复位写法，请确认这是设计意图，并检查复位释放时序。",
                    "sync_reset_review",
                )

        if not findings:
            findings.append({"severity": "pass", "message": "基础静态审查未发现明显 IC 设计常见问题。"})

        if len(verilog_code) < 80 or "module" not in code_lower:
            review_flags.append("input_may_be_incomplete")

        summary = "；".join(item["message"] for item in findings[:4])
        return {
            "summary": summary,
            "findings": findings,
            "evidence": evidence,
            "confidence": "medium" if review_flags else "high",
            "review_flags": sorted(set(review_flags)),
            "line_count": len(lines),
            "hint": "如需更深度分析，请提供完整模块代码和目标工艺/综合约束。",
        }


class TimingConstraintSuggesterTool(BaseTool):
    """时序约束建议工具（迁移自 rag_core.py）。"""

    def __init__(self) -> None:
        super().__init__()
        self.name = "timing_constraint_suggester"
        self.description = "时序约束建议工具（SDC）"
        self.risk_level = "high"
        self.parameters = [
            ToolParameter(name="module_name", type="string", description="模块名", required=False),
            ToolParameter(name="clock_period_ns", type="number", description="时钟周期(ns)", required=False),
            ToolParameter(name="io_description", type="string", description="IO描述", required=False),
            ToolParameter(name="query", type="string", description="原始用户问题", required=False),
        ]

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        query = str(kwargs.get("query", "")).strip()
        module_name = str(kwargs.get("module_name") or "user_module").strip() or "user_module"

        clock_period_ns_raw = kwargs.get("clock_period_ns")
        try:
            clock_period_ns = float(clock_period_ns_raw) if clock_period_ns_raw is not None else _guess_clock_period_ns(query, 5.0)
        except (TypeError, ValueError):
            clock_period_ns = _guess_clock_period_ns(query, 5.0)

        if clock_period_ns <= 0:
            clock_period_ns = 5.0

        io_description = str(kwargs.get("io_description") or query or "")
        review_flags: list[str] = []
        assumptions = [
            "默认主时钟端口名为 clk。",
            "默认复位端口 rst_n 是异步控制信号。",
            "IO 延迟比例为经验模板，需结合板级/上级模块时序确认。",
        ]

        sdc = f"""# ==================== 自动生成的SDC时序约束 ====================
# 模块: {module_name}
# 时钟周期: {clock_period_ns} ns ({1 / clock_period_ns * 1000:.1f} MHz)

# 1. 定义主时钟
create_clock -name sys_clk -period {clock_period_ns} [get_ports clk]

# 2. 时钟不确定性
set_clock_uncertainty 0.2 [get_clocks sys_clk]
set_clock_latency 0.1 [get_clocks sys_clk]

# 3. IO延迟约束（根据你提供的io_description智能调整）
"""

        if "input" in io_description.lower() or "in" in io_description.lower():
            sdc += f"set_input_delay -clock sys_clk -max {clock_period_ns * 0.3:.2f} [get_ports {{输入端口列表}}]\n"
            sdc += f"set_input_delay -clock sys_clk -min {clock_period_ns * 0.05:.2f} [get_ports {{输入端口列表}}]\n"
        else:
            review_flags.append("missing_input_delay_context")

        if "output" in io_description.lower() or "out" in io_description.lower():
            sdc += f"set_output_delay -clock sys_clk -max {clock_period_ns * 0.3:.2f} [get_ports {{输出端口列表}}]\n"
            sdc += f"set_output_delay -clock sys_clk -min {clock_period_ns * 0.05:.2f} [get_ports {{输出端口列表}}]\n"
        else:
            review_flags.append("missing_output_delay_context")

        sdc += "\n# 4. 其他常用约束（根据实际项目补充）\n"
        sdc += "set_false_path -from [get_ports rst_n]\n"
        sdc += "# set_multicycle_path -setup 2 -to [get_ports critical_output]\n"

        if module_name == "user_module":
            review_flags.append("module_name_defaulted")
        if clock_period_ns_raw is None and _extract_clock_period_from_text(query) is None:
            review_flags.append("clock_period_defaulted")

        return {
            "summary": f"已生成 {module_name} 的 SDC 模板，时钟周期 {clock_period_ns} ns。",
            "generated_sdc": sdc,
            "assumptions": assumptions,
            "evidence": [
                {
                    "type": "user_goal",
                    "source": "user_input",
                    "content": query or io_description,
                }
            ],
            "confidence": "low" if review_flags else "medium",
            "review_flags": sorted(set(review_flags)),
        }


def _extract_always_blocks(code: str) -> list[tuple[int, str, str]]:
    lines = code.splitlines()
    starts: list[int] = []
    for idx, line in enumerate(lines):
        if re.search(r"\balways(?:_comb|_ff)?\b", line):
            starts.append(idx)

    blocks: list[tuple[int, str, str]] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        header = lines[start].strip()
        body = "\n".join(lines[start:end])
        blocks.append((start + 1, header, body))
    return blocks


def _extract_sensitivity(header: str) -> str:
    match = re.search(r"@\s*\((.*?)\)", header or "")
    return match.group(1).lower() if match else ""


def _extract_clock_period_from_text(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*ns", (text or "").lower())
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None
