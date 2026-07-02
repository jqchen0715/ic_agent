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

        task.final_answer = await self._finalize(goal, task.steps, model_preference)
        task.reflection = await self._reflect(goal, task.final_answer, task.steps, model_preference)
        task.audit_summary = self._build_audit_summary(task.steps)
        task.review_flags = list(task.audit_summary.get("review_flags", []))
        task.confidence = str(task.audit_summary.get("confidence", "unknown"))
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
                return steps[:max_steps]
        except Exception as exc:  # noqa: BLE001
            logger.warning("自主 Agent 规划降级为启发式计划: {}", exc)

        return self._heuristic_plan(goal)[:max_steps]

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
            return {"query": f"{goal}\n{step.description}"}
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
    ) -> str:
        context = _steps_context(steps, max_len=14000)
        try:
            resp = await self._model_router.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是强自主 Agent 的最终交付器。输出应包含: 结论、执行过程、"
                            "可采纳建议、风险/证据边界、下一步。不要编造未观察到的来源。"
                            "若 review_flags 或 low confidence 存在，必须明确建议人工复核。"
                        ),
                    },
                    {"role": "user", "content": f"目标:\n{goal}\n\n执行记录:\n{context}"},
                ],
                model_preference=model_preference,
                temperature=0.2,
            )
            text = str(getattr(resp, "content", "") or "").strip()
            if text:
                return text
        except Exception as exc:  # noqa: BLE001
            logger.warning("自主 Agent 最终汇总降级: {}", exc)

        lines = ["# 自主任务执行结果", "", f"目标: {goal}", "", "## 执行轨迹"]
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
        lines.append("")
        lines.append("## 关键观察")
        for step in steps:
            observation = (step.observation or "").strip()
            if not observation:
                continue
            clipped = observation[:1200] + ("..." if len(observation) > 1200 else "")
            lines.append(f"### {step.title}")
            lines.append(clipped)
            lines.append("")
        lines.extend(
            [
                "## 结论",
                "Agent 已按计划完成可用步骤。带 low confidence、工具失败或复核标记的内容不能直接视为最终事实，"
                "需要人工确认或修复依赖后重新运行。",
                "",
                "## 下一步",
                "- 若需要知识库证据，请确认 embedding 模型路径和 Chroma 数据库可用。",
                "- 若需要更精确 SDC，请补充模块端口、时钟周期、复位、IO 延迟和多周期/伪路径约束背景。",
            ]
        )
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

    def _build_audit_summary(self, steps: list[AutonomousTaskStep]) -> dict[str, Any]:
        flags: list[str] = []
        tool_steps = [step for step in steps if step.action_type == "tool"]
        evidence_count = 0
        low_confidence = 0
        for step in steps:
            flags.extend(step.review_flags)
            evidence_count += len(step.evidence)
            if step.confidence in {"low", "unknown"}:
                low_confidence += 1

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
        q = goal.lower()
        terms = ("verilog", "rtl", "sdc", "setup", "hold", "时序", "芯片", "综合", "仿真")
        return any(term in q or term in goal for term in terms)

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
