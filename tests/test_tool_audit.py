# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest

from app.core.tools.factory import build_ic_tool_registry


@pytest.mark.asyncio
async def test_tool_registry_rejects_missing_required_argument():
    registry = build_ic_tool_registry()

    with pytest.raises(ValueError, match="verilog_code"):
        await registry.invoke_with_audit("verilog_code_analyzer", {})


@pytest.mark.asyncio
async def test_verilog_analyzer_returns_evidence_and_review_flags():
    registry = build_ic_tool_registry()
    code = """module latch(input a, input sel, output reg y);
always @(*) begin
  if (sel) y = a;
end
endmodule"""

    audit = await registry.invoke_with_audit("verilog_code_analyzer", {"verilog_code": code})

    assert audit["ok"] is True
    assert audit["confidence"] == "medium"
    assert "possible_latch" in audit["review_flags"]
    assert audit["evidence"]
    assert "latch" in audit["summary"].lower()


@pytest.mark.asyncio
async def test_timing_suggester_marks_missing_context_for_review():
    registry = build_ic_tool_registry()

    audit = await registry.invoke_with_audit(
        "timing_constraint_suggester",
        {"query": "给 2ns setup/hold 时序约束建议"},
    )

    assert audit["confidence"] == "low"
    assert "missing_input_delay_context" in audit["review_flags"]
    assert "missing_output_delay_context" in audit["review_flags"]
    assert audit["evidence"]
