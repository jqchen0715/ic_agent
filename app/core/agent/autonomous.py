# -*- coding: utf-8 -*-
"""强自主 Agent：目标规划、循环执行、反思审查与任务产物汇总。"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

from loguru import logger

from app.core.agent.reflection import ReflectionAgent, ReflectionReport
from app.core.intent import DomainClassification, ICDomainClassifier
from app.core.memory.manager import MemoryManager
from app.models.enums import MessageRole
from app.models.schemas import AutonomousTask, AutonomousTaskStep, Message


class ModelRouterLike(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        model_preference: str | None = None,
        **kwargs: Any,
    ) -> Any:
        ...


class ToolRegistryLike(Protocol):
    def list_tool_names(self) -> list[str]:
        ...

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        ...

    async def invoke_with_audit(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ...


class _ReflectionLLMAdapter:
    def __init__(self, model_router: ModelRouterLike, model_preference: str | None = None) -> None:
        self._model_router = model_router
        self._model_preference = model_preference

    async def acomplete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        resp = await self._model_router.chat(
            [dict(item) for item in messages],
            model_preference=self._model_preference,
            **kwargs,
        )
        return str(getattr(resp, "content", "") or "")


class AutonomousAgent:
    """面向 IC/Verilog 任务的强自主 Agent。"""

    def __init__(
        self,
        *,
        model_router: ModelRouterLike,
        tool_registry: ToolRegistryLike,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self._model_router = model_router
        self._tools = tool_registry
        self._memory = memory_manager
        self._domain_classifier = ICDomainClassifier()

    async def run(
        self,
        *,
        goal: str,
        session_id: str,
        model_preference: str | None = None,
        max_steps: int = 6,
    ) -> AutonomousTask:
        """执行完整自主任务闭环。"""
        task = AutonomousTask(
            id=str(uuid.uuid4()),
            session_id=session_id,
            goal=goal,
            status="running",
            created_at=_now(),
            updated_at=_now(),
            steps=[],
        )

        await self._remember(session_id, MessageRole.USER, goal, {"agent_mode": "autonomous"})

        plan = await self._plan(goal, model_preference, max_steps=max_steps)
        task.steps = plan
        task.updated_at = _now()

        for index, step in enumerate(task.steps):
            step.status = "running"
            step.started_at = _now()
            step.rationale = step.rationale or self._step_rationale(goal, step)
            task.updated_at = _now()
            try:
                if step.action_type == "tool" and step.tool_name:
                    args = self._build_tool_args(step.tool_name, goal, step)
                    step.arguments = args
                    audit = await self._invoke_tool_with_audit(step.tool_name, args)
                    step.observation = str(audit.get("result", ""))
                    step.evidence = _as_evidence_list(audit.get("evidence"))
                    step.confidence = _normalize_confidence(audit.get("confidence"))
                    step.review_flags = _as_str_list(audit.get("review_flags"))
                    if not bool(audit.get("ok", True)):
                        step.error = str(audit.get("summary") or "工具返回失败状态")
                else:
                    step.observation = await self._reason(goal, task.steps[:index], step, model_preference)
                    step.confidence = "low"
                    step.review_flags = sorted(set([*step.review_flags, "reasoning_without_tool_evidence"]))
                step.status = "completed"
            except Exception as exc:  # noqa: BLE001
                logger.exception("自主 Agent 子任务失败 step={}", step.id)
                step.status = "failed"
                step.error = str(exc)
                step.confidence = "low"
                step.review_flags = sorted(set([*step.review_flags, "tool_execution_failed"]))
                if self._should_recover(step):
                    recovered = await self._recover(goal, step, model_preference)
                    step.observation = (
                        "工具执行失败，已切换为降级推理。\n"
                        f"原始错误: {exc}\n\n"
                        f"{recovered}"
                    )
                    step.status = "completed"
                    step.review_flags = sorted(set([*step.review_flags, "recovered_with_reasoning"]))
                else:
                    task.status = "failed"
                    task.error = str(exc)
                    break
            finally:
                step.finished_at = _now()

        task.audit_summary = self._build_audit_summary(task.steps)
        task.review_flags = list(task.audit_summary.get("review_flags", []))
        task.confidence = str(task.audit_summary.get("confidence", "unknown"))
        task.answer_mode = self._derive_answer_mode(task.audit_summary)
        sections = self._build_answer_sections(goal, task.steps, task.audit_summary, task.answer_mode)
        task.evidence_supported = sections["evidence_supported"]
        task.draft_suggestions = sections["draft_suggestions"]
        task.missing_evidence = sections["missing_evidence"]
        task.next_actions = sections["next_actions"]
        task.final_answer = await self._finalize(
            goal,
            task.steps,
            model_preference,
            audit_summary=task.audit_summary,
            answer_mode=task.answer_mode,
            sections=sections,
        )
        task.reflection = await self._reflect(goal, task.final_answer, task.steps, model_preference)
        if task.answer_mode == "strict_answer" and self._reflection_requires_review(task.reflection):
            task.answer_mode = "assisted_draft"
            task.review_flags = sorted(set([*task.review_flags, "reflection_requires_review"]))
            task.audit_summary["review_flags"] = task.review_flags
            task.audit_summary["confidence"] = "low"
            task.confidence = "low"
            sections = self._build_answer_sections(goal, task.steps, task.audit_summary, task.answer_mode)
            task.evidence_supported = sections["evidence_supported"]
            task.draft_suggestions = sections["draft_suggestions"]
            task.missing_evidence = sections["missing_evidence"]
            task.next_actions = sections["next_actions"]
            task.final_answer = await self._finalize(
                goal,
                task.steps,
                model_preference,
                audit_summary=task.audit_summary,
                answer_mode=task.answer_mode,
                sections=sections,
            )
        task.status = self._derive_status(task)
        task.updated_at = _now()

        await self._remember(
            session_id,
            MessageRole.ASSISTANT,
            task.final_answer or "",
            {"agent_mode": "autonomous", "task_id": task.id, "status": task.status},
        )
        return task

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

    async def _plan(
        self,
        goal: str,
        model_preference: str | None,
        *,
        max_steps: int,
    ) -> list[AutonomousTaskStep]:
        system = (
            "你是 IC/Verilog 强自主 Agent 的规划器。"
            "请将用户目标拆成 2-6 个可执行步骤，并选择必要工具。"
            "可用工具: ic_rag_search, verilog_code_analyzer, timing_constraint_suggester。"
            "只输出 JSON: {\"steps\":[{\"title\":\"...\",\"description\":\"...\","
            "\"action_type\":\"tool|reasoning\",\"tool_name\":\"工具名或null\","
            "\"rationale\":\"为什么需要这一步\"}]}"
        )
        try:
            resp = await self._model_router.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": goal},
                ],
                model_preference=model_preference,
                temperature=0.2,
            )
            payload = _extract_json(str(getattr(resp, "content", "") or ""))
            steps = _parse_steps(payload)
            if steps:
                scope = await self._domain_classifier.classify(
                    goal,
                    model_router=self._model_router,
                    model_preference=model_preference,
                )
                return self._ensure_scope_retrieval_step(goal, steps, scope)[:max_steps]
        except Exception as exc:  # noqa: BLE001
            logger.warning("自主 Agent 规划降级为启发式计划: {}", exc)

        return self._heuristic_plan(goal)[:max_steps]

    def _ensure_scope_retrieval_step(
        self,
        goal: str,
        steps: list[AutonomousTaskStep],
        scope: DomainClassification,
    ) -> list[AutonomousTaskStep]:
        if not scope.should_retrieve:
            return steps
        if any(step.tool_name == "ic_rag_search" for step in steps):
            return steps

        retrieval_step = AutonomousTaskStep(
            id="scope_rag",
            title="检索 IC 知识库证据",
            description=(
                f"基于领域分类结果检索相关片段。domain={scope.domain}; "
                f"query={scope.normalized_query or goal}"
            ),
            action_type="tool",
            tool_name="ic_rag_search",
            rationale=(
                "领域分类器判断该目标应进入 IC/Verilog 知识库检索，"
                "后续结论需要受检索证据约束。"
            ),
        )
        insert_at = 1 if steps and steps[0].action_type == "reasoning" else 0
        return [*steps[:insert_at], retrieval_step, *steps[insert_at:]]

    def _heuristic_plan(self, goal: str) -> list[AutonomousTaskStep]:
        q = goal.lower()
        steps: list[AutonomousTaskStep] = [
            AutonomousTaskStep(
                id="s1",
                title="澄清目标与约束",
                description="识别用户目标、输入材料、输出产物和可能需要的工具。",
                action_type="reasoning",
                rationale="先确认目标边界，避免直接给出无证据结论。",
            )
        ]
        if self._is_ic_query(goal):
            steps.append(
                AutonomousTaskStep(
                    id=f"s{len(steps) + 1}",
                    title="检索 IC 知识库证据",
                    description="检索与目标相关的知识库片段，作为后续结论依据。",
                    action_type="tool",
                    tool_name="ic_rag_search",
                    rationale="IC 知识问答需要知识库片段作为可引用证据。",
                )
            )
        if self._looks_like_verilog(goal):
            steps.append(
                AutonomousTaskStep(
                    id=f"s{len(steps) + 1}",
                    title="审查 Verilog/RTL 风险",
                    description="检查代码或描述中的 RTL 编码、综合和仿真风险。",
                    action_type="tool",
                    tool_name="verilog_code_analyzer",
                    rationale="用户目标包含 Verilog/RTL 信号，适合先做静态规则审查。",
                )
            )
        if any(token in q or token in goal for token in ("sdc", "setup", "hold", "时序", "约束")):
            steps.append(
                AutonomousTaskStep(
                    id=f"s{len(steps) + 1}",
                    title="生成时序约束建议",
                    description="根据目标生成 SDC/时序约束与优化方向。",
                    action_type="tool",
                    tool_name="timing_constraint_suggester",
                    rationale="目标涉及 setup/hold/SDC/时序约束，需要生成可复核约束草案。",
                )
            )
        steps.append(
            AutonomousTaskStep(
                id=f"s{len(steps) + 1}",
                title="整合执行结果",
                description="汇总工具观察、失败原因、证据边界和最终建议。",
                action_type="reasoning",
                rationale="将工具证据、置信度和复核点合并为最终交付。",
            )
        )
        return steps

    def _step_rationale(self, goal: str, step: AutonomousTaskStep) -> str:
        if step.action_type == "tool" and step.tool_name:
            return f"为目标“{goal[:80]}”收集 {step.tool_name} 的可审计输出。"
        return "对已有输入和工具观察做阶段性整理。"

    async def _reason(
        self,
        goal: str,
        previous_steps: list[AutonomousTaskStep],
        step: AutonomousTaskStep,
        model_preference: str | None,
    ) -> str:
        context = _steps_context(previous_steps)
        try:
            resp = await self._model_router.chat(
                [
                    {
                        "role": "system",
                        "content": "你是自主 Agent 的执行器。基于目标和已有观察完成当前步骤。",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"目标:\n{goal}\n\n当前步骤:\n{step.title}\n{step.description}"
                            f"\n\n已有观察:\n{context}"
                        ),
                    },
                ],
                model_preference=model_preference,
                temperature=0.2,
            )
            return str(getattr(resp, "content", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("自主 Agent 推理步骤降级: {}", exc)
            if previous_steps:
                return "已基于前序工具观察进入整合阶段；模型推理暂不可用，保留原始观察供最终汇总。"
            return "已识别任务目标，后续将优先调用可用工具收集证据。"

    def _build_tool_args(
        self,
        tool_name: str,
        goal: str,
        step: AutonomousTaskStep,
    ) -> dict[str, Any]:
        if tool_name == "ic_rag_search":
            scope = self._domain_classifier.classify_by_rules(goal)
            query = scope.normalized_query if scope is not None else goal
            return {"query": f"{query}\n{step.description}"}
        if tool_name == "verilog_code_analyzer":
            code = _extract_code(goal) or goal
            return {"verilog_code": code}
        if tool_name == "timing_constraint_suggester":
            args: dict[str, Any] = {"query": goal}
            module_name = _extract_module_name(goal)
            period = _extract_clock_period(goal)
            if module_name:
                args["module_name"] = module_name
            if period is not None:
                args["clock_period_ns"] = period
            return args
        return {"query": goal}

    def _should_recover(self, step: AutonomousTaskStep) -> bool:
        return step.action_type == "tool"

    async def _recover(
        self,
        goal: str,
        step: AutonomousTaskStep,
        model_preference: str | None,
    ) -> str:
        return await self._reason(goal, [], step, model_preference)

    async def _finalize(
        self,
        goal: str,
        steps: list[AutonomousTaskStep],
        model_preference: str | None,
        *,
        audit_summary: dict[str, Any],
        answer_mode: str,
        sections: dict[str, list[str]],
    ) -> str:
        context = _steps_context(steps, max_len=14000)
        section_context = json.dumps(sections, ensure_ascii=False, indent=2)
        audit_context = json.dumps(audit_summary, ensure_ascii=False, indent=2)
        try:
            resp = await self._model_router.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是 IC/Verilog 自主任务的最终交付器。必须严格按以下四个二级标题输出: "
                            "证据支持、草案建议、缺失证据、下一步。"
                            "answer_mode=strict_answer 时，只能把工具证据支持的内容写成确定表述。"
                            "answer_mode=assisted_draft 时，必须说明这是需人工复核的辅助草案，"
                            "不得把无证据内容写成确定结论。answer_mode=refusal 时，只说明无法交付和需要补充什么。"
                            "不要编造未观察到的来源、页码、工具结果或知识库证据。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"目标:\n{goal}\n\nanswer_mode:\n{answer_mode}\n\n审计摘要:\n"
                            f"{audit_context}\n\n结构化交付要点:\n{section_context}\n\n执行记录:\n{context}"
                        ),
                    },
                ],
                model_preference=model_preference,
                temperature=0.2,
            )
            text = str(getattr(resp, "content", "") or "").strip()
            if text:
                return text
        except Exception as exc:  # noqa: BLE001
            logger.warning("自主 Agent 最终汇总降级: {}", exc)

        return self._render_structured_final_answer(goal, answer_mode, sections, audit_summary, steps)

    def _render_structured_final_answer(
        self,
        goal: str,
        answer_mode: str,
        sections: dict[str, list[str]],
        audit_summary: dict[str, Any],
        steps: list[AutonomousTaskStep],
    ) -> str:
        lines = [
            "# 自主任务执行结果",
            "",
            f"目标: {goal}",
            f"模式: {answer_mode}",
            f"置信度: {audit_summary.get('confidence', 'unknown')}",
            "",
            "## 证据支持",
        ]
        lines.extend(_format_bullets(sections.get("evidence_supported") or ["无可引用工具证据。"]))
        lines.extend(["", "## 草案建议"])
        lines.extend(_format_bullets(sections.get("draft_suggestions") or ["暂无可安全交付的草案建议。"]))
        lines.extend(["", "## 缺失证据"])
        lines.extend(_format_bullets(sections.get("missing_evidence") or ["暂无额外缺失证据。"]))
        lines.extend(["", "## 下一步"])
        lines.extend(_format_bullets(sections.get("next_actions") or ["补充输入或修复依赖后重新运行。"]))
        lines.extend(["", "## 执行轨迹"])
        for step in steps:
            status = "完成" if step.status == "completed" else "失败"
            lines.append(f"- {step.title}: {status}")
            lines.append(f"  - 置信度: {step.confidence}")
            if step.review_flags:
                lines.append(f"  - 复核标记: {', '.join(step.review_flags)}")
            if step.evidence:
                lines.append(f"  - 证据数: {len(step.evidence)}")
            if step.error:
                lines.append(f"  - 边界: {step.error}")
        return "\n".join(lines)

    async def _reflect(
        self,
        goal: str,
        final_answer: str,
        steps: list[AutonomousTaskStep],
        model_preference: str | None,
    ) -> dict[str, Any]:
        evidence = [
            str(step.observation or "")[:1200]
            for step in steps
            if step.observation and step.action_type == "tool"
        ]
        try:
            report = await ReflectionAgent(
                _ReflectionLLMAdapter(self._model_router, model_preference),
                min_quality_to_pass=60,
            ).reflect(
                goal,
                final_answer,
                evidence_snippets=evidence,
                trace_summary=_steps_context(steps, max_len=4000),
            )
            return _reflection_to_dict(report)
        except Exception as exc:  # noqa: BLE001
            logger.warning("自主 Agent 反思降级: {}", exc)
            failed = [step for step in steps if step.status == "failed"]
            return {
                "quality_score": 60 if not failed else 45,
                "is_complete": not failed,
                "likely_hallucination": False,
                "summary": "反思模型不可用，已基于步骤状态给出保守审查。",
                "suggestions": ["请人工复核工具失败步骤"] if failed else [],
            }

    def _derive_status(self, task: AutonomousTask) -> str:
        if task.error or any(step.status == "failed" for step in task.steps):
            return "failed"
        if any(step.error for step in task.steps):
            return "needs_review"
        if task.review_flags:
            return "needs_review"
        if task.confidence in {"low", "unknown"}:
            return "needs_review"
        reflection = task.reflection or {}
        if reflection.get("likely_hallucination"):
            return "needs_review"
        if reflection.get("quality_score", 100) < 60:
            return "needs_review"
        return "completed"

    def _derive_answer_mode(self, audit_summary: dict[str, Any]) -> str:
        evidence_count = _safe_int(audit_summary.get("evidence_count"))
        failed_count = _safe_int(audit_summary.get("failed_step_count"))
        confidence = str(audit_summary.get("confidence", "unknown")).lower()
        review_flags = _as_str_list(audit_summary.get("review_flags"))
        if failed_count and not evidence_count:
            return "refusal"
        if evidence_count and not review_flags and confidence in {"high", "medium"}:
            return "strict_answer"
        return "assisted_draft"

    def _reflection_requires_review(self, reflection: dict[str, Any]) -> bool:
        if reflection.get("likely_hallucination"):
            return True
        if _safe_int(reflection.get("quality_score", 100)) < 60:
            return True
        return bool(reflection.get("parse_error"))

    def _build_answer_sections(
        self,
        goal: str,
        steps: list[AutonomousTaskStep],
        audit_summary: dict[str, Any],
        answer_mode: str,
    ) -> dict[str, list[str]]:
        evidence_supported = self._collect_evidence_supported(steps)
        review_flags = _as_str_list(audit_summary.get("review_flags"))
        missing_evidence = self._missing_evidence_items(review_flags, steps, evidence_supported)
        next_actions = self._next_action_items(goal, review_flags, answer_mode)

        if answer_mode == "strict_answer":
            draft_suggestions = self._collect_tool_suggestions(steps)
            if not draft_suggestions:
                draft_suggestions = ["基于已命中的工具证据整理结论；若用于面试或设计评审，仍建议核对原始文档页码。"]
        elif answer_mode == "refusal":
            draft_suggestions = ["当前任务没有形成可交付草案；需要先修复失败步骤或补充输入。"]
        else:
            draft_suggestions = self._collect_draft_suggestions(steps)
            if not draft_suggestions:
                draft_suggestions = [
                    "这是辅助草案，不是知识库证据支持的确定答案。",
                    "先明确目标规格、输入输出约束、性能目标和已有设计材料，再重新触发检索与工具审查。",
                ]

        if not evidence_supported:
            evidence_supported = ["无可引用工具证据；以下内容只能作为需人工复核的辅助草案。"]

        return {
            "evidence_supported": evidence_supported[:8],
            "draft_suggestions": draft_suggestions[:8],
            "missing_evidence": missing_evidence[:8],
            "next_actions": next_actions[:8],
        }

    def _collect_evidence_supported(self, steps: list[AutonomousTaskStep]) -> list[str]:
        items: list[str] = []
        for step in steps:
            for evidence in step.evidence:
                content = str(evidence.get("content") or evidence.get("summary") or "").strip()
                if not content:
                    continue
                source = str(evidence.get("source") or "").strip()
                page = evidence.get("page")
                ref = f"（{source} 第{page}页）" if source and page not in (None, "", "未知") else ""
                items.append(f"{step.title}: {_clip(content, 220)}{ref}")
        return items

    def _collect_tool_suggestions(self, steps: list[AutonomousTaskStep]) -> list[str]:
        items: list[str] = []
        for step in steps:
            if step.action_type != "tool" or not step.observation or step.review_flags:
                continue
            items.append(f"{step.title}: {_clip(step.observation, 260)}")
        return items

    def _collect_draft_suggestions(self, steps: list[AutonomousTaskStep]) -> list[str]:
        items: list[str] = []
        for step in steps:
            observation = (step.observation or "").strip()
            if not observation:
                continue
            if step.evidence and not step.review_flags:
                continue
            label = "工具观察" if step.action_type == "tool" else "推理草案"
            items.append(f"{label} - {step.title}: {_clip(observation, 260)}")
        return items

    def _missing_evidence_items(
        self,
        review_flags: list[str],
        steps: list[AutonomousTaskStep],
        evidence_supported: list[str],
    ) -> list[str]:
        items: list[str] = []
        if not evidence_supported:
            items.append("当前没有可引用知识库片段或工具证据。")
        for flag in review_flags:
            if flag == "rag_step_without_evidence":
                items.append("RAG 检索步骤没有返回可引用证据。")
            elif flag in {"rag_no_results", "rag_weak_evidence"}:
                items.append("知识库检索结果为空或相关性不足。")
            elif flag == "reasoning_without_tool_evidence":
                items.append("部分步骤依赖模型推理，未绑定工具证据。")
            elif flag == "recovered_with_reasoning":
                items.append("有步骤从工具失败降级为推理恢复。")
            elif flag.endswith("_error") or flag == "tool_execution_failed":
                items.append(f"工具执行存在失败标记: {flag}。")
            elif flag == "reflection_requires_review":
                items.append("反思审查未通过或不可用，最终交付需要人工复核。")
            else:
                items.append(f"复核标记: {flag}。")
        for step in steps:
            if step.error:
                items.append(f"{step.title} 失败或异常: {_clip(step.error, 180)}")
        return sorted(set(items)) or ["暂无额外缺失证据。"]

    def _next_action_items(self, goal: str, review_flags: list[str], answer_mode: str) -> list[str]:
        items: list[str] = []
        if answer_mode != "strict_answer":
            items.append("把草案内容当作待复核工作单，不要直接当作最终答案。")
        if "rag_step_without_evidence" in review_flags or "rag_no_results" in review_flags:
            items.append("补充或重建相关 IC/Verilog PDF 知识库后重新运行检索。")
        if any(flag.endswith("_error") or flag == "tool_execution_failed" for flag in review_flags):
            items.append("先修复失败工具依赖，再重新执行自主任务。")
        if any(token in goal for token in ("乘法器", "加法器", "FIFO", "状态机", "时序", "约束")):
            items.append("补充位宽、时钟周期、延迟目标、面积/功耗约束和目标平台。")
        items.append("将最终结论回到普通问答或检索证据链中复核。")
        return items

    def _build_audit_summary(self, steps: list[AutonomousTaskStep]) -> dict[str, Any]:
        flags: list[str] = []
        tool_steps = [step for step in steps if step.action_type == "tool"]
        evidence_count = 0
        low_confidence = 0
        failed_count = 0
        for step in steps:
            flags.extend(step.review_flags)
            evidence_count += len(step.evidence)
            if step.confidence in {"low", "unknown"}:
                low_confidence += 1
            if step.status == "failed":
                failed_count += 1

        for step in tool_steps:
            if not step.evidence and step.tool_name == "ic_rag_search":
                flags.append("rag_step_without_evidence")
            if step.error:
                flags.append(f"{step.tool_name or 'tool'}_error")

        unique_flags = sorted(set(flag for flag in flags if flag))
        if any(step.status == "failed" for step in steps) or any(step.error for step in steps):
            confidence = "low"
        elif unique_flags:
            confidence = "low"
        elif low_confidence:
            confidence = "medium"
        elif evidence_count:
            confidence = "high"
        else:
            confidence = "unknown"

        return {
            "confidence": confidence,
            "review_flags": unique_flags,
            "evidence_count": evidence_count,
            "tool_step_count": len(tool_steps),
            "low_confidence_step_count": low_confidence,
            "failed_step_count": failed_count,
        }

    async def _remember(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        if self._memory is None or not content.strip():
            return
        message = Message(role=role, content=content, metadata=metadata)
        try:
            await self._memory.save(session_id, message)
            await self._memory.remember(session_id, message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("自主 Agent 写入记忆失败: {}", exc)

    def _is_ic_query(self, goal: str) -> bool:
        return self._domain_classifier.classify_by_rules(goal) is not None

    def _looks_like_verilog(self, goal: str) -> bool:
        q = goal.lower()
        return bool(re.search(r"\b(module|endmodule|always|assign)\b", q)) or "```" in goal


class AgentTaskStore:
    """进程内任务存储，适合本地演示和同步 API。"""

    def __init__(self) -> None:
        self._items: dict[str, AutonomousTask] = {}

    def save(self, task: AutonomousTask) -> None:
        self._items[task.id] = task

    def get(self, task_id: str) -> AutonomousTask | None:
        return self._items.get(task_id)

    def list_recent(self, limit: int = 20) -> list[AutonomousTask]:
        items = sorted(self._items.values(), key=lambda item: item.created_at, reverse=True)
        return items[: max(1, limit)]


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _parse_steps(payload: dict[str, Any]) -> list[AutonomousTaskStep]:
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        return []
    steps: list[AutonomousTaskStep] = []
    allowed_tools = {"ic_rag_search", "verilog_code_analyzer", "timing_constraint_suggester"}
    for index, item in enumerate(raw_steps, 1):
        if not isinstance(item, dict):
            continue
        action_type = str(item.get("action_type", "reasoning")).lower()
        tool_name = item.get("tool_name")
        if action_type != "tool" or tool_name not in allowed_tools:
            action_type = "reasoning"
            tool_name = None
        steps.append(
            AutonomousTaskStep(
                id=str(item.get("id") or f"s{index}"),
                title=str(item.get("title") or f"步骤 {index}"),
                description=str(item.get("description") or ""),
                action_type=action_type,
                tool_name=tool_name,
                rationale=str(item.get("rationale") or ""),
            )
        )
    return steps


def _as_evidence_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _normalize_confidence(value: Any) -> str:
    confidence = str(value or "unknown").lower()
    if confidence in {"high", "medium", "low", "unknown"}:
        return confidence
    return "unknown"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _clip(text: str, max_len: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def _format_bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items if item.strip()]


def _steps_context(steps: list[AutonomousTaskStep], max_len: int = 10000) -> str:
    data = [step.model_dump() for step in steps]
    return json.dumps(data, ensure_ascii=False, indent=2)[:max_len]


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:verilog|sv)?\s*([\s\S]*?)```", text or "", flags=re.I)
    return match.group(1).strip() if match else ""


def _extract_module_name(text: str) -> str | None:
    match = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)", text or "", flags=re.I)
    return match.group(1) if match else None


def _extract_clock_period(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*ns", (text or "").lower())
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


def _reflection_to_dict(report: ReflectionReport) -> dict[str, Any]:
    return {
        "quality_score": report.quality_score,
        "is_complete": report.is_complete,
        "likely_hallucination": report.likely_hallucination,
        "hallucination_reasons": report.hallucination_reasons,
        "completeness_notes": report.completeness_notes,
        "suggestions": report.suggestions,
        "summary": report.summary,
        "parse_error": report.parse_error,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
