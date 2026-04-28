# -*- coding: utf-8 -*-
"""对话 API：非流式与流式输出。"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app.config import get_settings
from app.core.agent.langgraph_agent import LangGraphICAgent
from app.core.rag.citation_rewriter import rewrite_answer_citations
from app.core.tools.factory import build_ic_tool_registry
from app.infrastructure.llm.model_router import ModelConfig, ModelRouter
from app.infrastructure.trace.tracer import Tracer
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

_tracer = Tracer()


def _build_router() -> ModelRouter:
    """根据配置构造模型路由器。"""
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="未配置 OPENAI_API_KEY，无法调用模型")
    cfg = ModelConfig(
        model_id=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base or None,
        priority=0,
        weight=1.0,
    )
    return ModelRouter([cfg])


def _build_ic_agent() -> LangGraphICAgent:
    """构造 IC LangGraph 主链路 Agent。"""
    return LangGraphICAgent(
        model_router=_build_router(),
        tool_registry=build_ic_tool_registry(),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """非流式对话：走 LangGraph 主链路（pre_tool_router -> tools -> final answer）。"""
    trace_id = str(uuid.uuid4())
    span = _tracer.start_trace(trace_id, "chat")

    try:
        messages = [m.model_dump() for m in request.messages]

        agent = _build_ic_agent()
        result = await agent.run(
            messages=messages,
            model_preference=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        rewritten = rewrite_answer_citations(result.content, result.sources)

        model_used = result.model_id or request.model or get_settings().openai_model
        logger.info(
            "LangGraph 路由完成 trace={} tools={} reason={} clarify={} removed_fake_refs={}",
            trace_id,
            result.selected_tools,
            result.route_reason,
            result.needs_clarification,
            rewritten.removed_fake_count,
        )

        _tracer.end_span(
            span,
            result={
                "model": model_used,
                "usage": result.usage,
                "tools": result.selected_tools,
                "tool_events": len(result.tool_events),
                "sources": len(result.sources),
                "route_reason": result.route_reason,
                "needs_clarification": result.needs_clarification,
                "removed_fake_refs": rewritten.removed_fake_count,
            },
        )

        return ChatResponse(
            id=str(uuid.uuid4()),
            model=model_used,
            answer=rewritten.answer,
            content=rewritten.answer,
            trace_id=trace_id,
            usage=result.usage,
            sources=result.sources,
            tool_events=result.tool_events,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chat 失败: {}", exc)
        _tracer.end_span(span, error=str(exc))
        raise HTTPException(status_code=500, detail=f"对话失败: {exc!s}") from exc


async def _stream_generator(
    request: ChatRequest,
    trace_id: str,
    span: object,
) -> AsyncIterator[bytes]:
    """SSE 风格流：每行 data: {json}\\n\\n。"""
    def _emit(event: str, payload: dict[str, Any]) -> bytes:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()

    def _summarize_tool_result(raw: Any, max_len: int = 240) -> str:
        text = " ".join(str(raw or "").strip().split())
        if not text:
            return "（空结果）"
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    try:
        messages = [m.model_dump() for m in request.messages]
        agent = _build_ic_agent()
        result = await agent.run(
            messages=messages,
            model_preference=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        rewritten = rewrite_answer_citations(result.content, result.sources)
        model_used = result.model_id or request.model or get_settings().openai_model

        for event in result.tool_events:
            tool_name = str(event.get("tool", "unknown_tool"))
            tool_args = event.get("arguments", {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            yield _emit(
                "tool_call",
                {
                    "trace_id": trace_id,
                    "tool": tool_name,
                    "arguments": tool_args,
                },
            )

            yield _emit(
                "tool_result",
                {
                    "trace_id": trace_id,
                    "tool": tool_name,
                    "ok": bool(event.get("ok", False)),
                    "summary": _summarize_tool_result(event.get("result", "")),
                },
            )

        content = rewritten.answer or ""
        chunk_size = 64
        for idx in range(0, len(content), chunk_size):
            delta = content[idx : idx + chunk_size]
            if delta:
                yield _emit(
                    "answer",
                    {
                        "trace_id": trace_id,
                        "chunk": delta,
                    },
                )

        yield _emit(
            "citation",
            {
                "trace_id": trace_id,
                "sources": rewritten.references,
            },
        )

        yield _emit(
            "done",
            {
                "trace_id": trace_id,
                "model": model_used,
                "usage": result.usage,
            },
        )

        _tracer.end_span(
            span,
            result={
                "mode": "stream",
                "model": model_used,
                "tool_events": len(result.tool_events),
                "sources": len(result.sources),
                "removed_fake_refs": rewritten.removed_fake_count,
            },
        )
    except Exception as exc:
        logger.exception("chat_stream 失败: {}", exc)
        _tracer.end_span(span, error=str(exc))
        yield _emit(
            "error",
            {
                "trace_id": trace_id,
                "error": str(exc),
            },
        )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """流式输出（Server-Sent Events 兼容格式）。"""
    trace_id = str(uuid.uuid4())
    span = _tracer.start_trace(trace_id, "chat_stream")

    return StreamingResponse(
        _stream_generator(request, trace_id, span),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Trace-Id": trace_id,
        },
    )
