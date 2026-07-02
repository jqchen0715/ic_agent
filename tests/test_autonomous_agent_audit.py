# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

import pytest

from app.core.agent.autonomous import AutonomousAgent


class _UnavailableModel:
    async def chat(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("model unavailable")


class _AuditedFakeTools:
    def list_tool_names(self) -> list[str]:
        return ["ic_rag_search", "timing_constraint_suggester"]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        return str((await self.invoke_with_audit(name, arguments))["result"])

    async def invoke_with_audit(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "ic_rag_search":
            return {
                "tool": name,
                "arguments": arguments,
                "result": '{"results": []}',
                "ok": True,
                "summary": "no evidence",
                "evidence": [],
                "confidence": "low",
                "review_flags": ["rag_no_results"],
            }

        return {
            "tool": name,
            "arguments": arguments,
            "result": "sdc template",
            "ok": True,
            "summary": "sdc template",
            "evidence": [{"type": "user_goal", "content": "2ns timing"}],
            "confidence": "medium",
            "review_flags": [],
        }


@pytest.mark.asyncio
async def test_autonomous_agent_requires_review_without_rag_evidence():
    agent = AutonomousAgent(
        model_router=_UnavailableModel(),
        tool_registry=_AuditedFakeTools(),
    )

    task = await agent.run(goal="分析 setup/hold 时序问题", session_id="s-test", max_steps=4)

    assert task.status == "needs_review"
    assert task.confidence == "low"
    assert "rag_no_results" in task.review_flags
    assert "rag_step_without_evidence" in task.review_flags
    assert task.audit_summary["evidence_count"] == 1
    assert all(step.rationale for step in task.steps)
