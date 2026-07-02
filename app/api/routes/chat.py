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
from app.core.memory.factory import get_memory_manager
from app.core.rag.citation_rewriter import rewrite_answer_citations
from app.core.tools.factory import build_ic_tool_registry
from app.infrastructure.llm.model_router import ModelConfig, ModelRouter
from app.infrastructure.trace.tracer import Tracer
from app.models.enums import MessageRole
from app.models.schemas import ChatMessage, ChatRequest, ChatResponse, MemoryContext, Message

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


def _latest_user_content(messages: list[ChatMessage]) -> str:
    for item in reversed(messages):
        if item.role.lower() == "user":
            return item.content
    return messages[-1].content if messages else ""


def _to_memory_message(message: ChatMessage, trace_id: str) -> Message | None:
    try:
        role = MessageRole(message.role.lower())
    except ValueError:
        return None
    return Message(
        role=role,
        content=message.content,
        metadata={"trace_id": trace_id, "source": "chat_api"},
    )


def _message_key(message: dict[str, Any]) -> tuple[str, str]:
    return (str(message.get("role", "")), str(message.get("content", "")))


def _merge_memory_messages(
    memory_messages: list[dict[str, Any]],
    base_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """合并记忆历史与请求消息，只消除两段之间的重叠。"""
    max_overlap = min(len(memory_messages), len(base_messages))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        memory_tail = [_message_key(item) for item in memory_messages[-size:]]
        base_head = [_message_key(item) for item in base_messages[:size]]
        if memory_tail == base_head:
            overlap = size
            break
    return [*memory_messages, *base_messages[overlap:]]


def _render_long_term_memory(context: MemoryContext) -> str:
    if not context.long_term_items:
        return ""
    lines = [
        "以下是同一会话召回的长期记忆，仅用于理解上下文；"
        "若与用户本轮输入冲突，以本轮输入为准。"
    ]
    for idx, item in enumerate(context.long_term_items, 1):
        role = str(item.metadata.get("role", "memory"))
        lines.append(f"[M{idx}] ({role}) {item.content}")
    return "\n".join(lines)


async def _prepare_messages(
    request: ChatRequest,
    session_id: str,
    trace_id: str,
) -> tuple[list[dict[str, Any]], MemoryContext | None]:
    """读取记忆并拼装给 Agent 的消息。"""
    base_messages = [m.model_dump() for m in request.messages]
    memory = get_memory_manager()
    if memory is None:
        return base_messages, None

    try:
        context = await memory.get_context(session_id, _latest_user_content(request.messages))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "读取记忆失败，已降级为无记忆对话 trace={}: {}",
            trace_id,
            exc,
        )
        return base_messages, None

    memory_messages: list[dict[str, Any]] = []
    long_term_text = _render_long_term_memory(context)
    if long_term_text:
        memory_messages.append({"role": "system", "content": long_term_text})

    for message in context.short_term_messages:
        if message.role in {MessageRole.USER, MessageRole.ASSISTANT, MessageRole.SYSTEM}:
            memory_messages.append(
                {
                    "role": message.role.value,
                    "content": message.content,
                }
            )

    return _merge_memory_messages(memory_messages, base_messages), context


async def _save_memory_turn(
    request: ChatRequest,
    session_id: str,
    trace_id: str,
    assistant_answer: str,
) -> None:
    memory = get_memory_manager()
    if memory is None:
        return

    user_msg: Message | None = None
    for item in reversed(request.messages):
        if item.role.lower() == "user":
            user_msg = _to_memory_message(item, trace_id)
            break

    assistant_msg = Message(
        role=MessageRole.ASSISTANT,
        content=assistant_answer,
        metadata={"trace_id": trace_id, "source": "chat_api"},
    )

    try:
        if user_msg is not None:
            await memory.save(session_id, user_msg)
            await memory.remember(session_id, user_msg)
        await memory.save(session_id, assistant_msg)
        await memory.remember(session_id, assistant_msg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("保存记忆失败，已忽略 trace={}: {}", trace_id, exc)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """非流式对话：走 LangGraph 主链路（pre_tool_router -> tools -> final answer）。"""
    trace_id = str(uuid.uuid4())
    span = _tracer.start_trace(trace_id, "chat")
    session_id = request.conversation_id or str(uuid.uuid4())

    try:
        messages, memory_context = await _prepare_messages(request, session_id, trace_id)

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
        await _save_memory_turn(request, session_id, trace_id, rewritten.answer)

        _tracer.end_span(
            span,
            result={
                "conversation_id": session_id,
                "model": model_used,
                "usage": result.usage,
                "tools": result.selected_tools,
                "tool_events": len(result.tool_events),
                "sources": len(result.sources),
                "route_reason": result.route_reason,
                "needs_clarification": result.needs_clarification,
                "removed_fake_refs": rewritten.removed_fake_count,
                "memory_short_items": (
                    len(memory_context.short_term_messages) if memory_context else 0
                ),
                "memory_long_items": len(memory_context.long_term_items) if memory_context else 0,
            },
        )

        return ChatResponse(
            id=str(uuid.uuid4()),
            conversation_id=session_id,
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
        session_id = request.conversation_id or str(uuid.uuid4())
        messages, memory_context = await _prepare_messages(request, session_id, trace_id)
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
                    "summary": event.get("summary") or _summarize_tool_result(event.get("result", "")),
                    "confidence": event.get("confidence", "unknown"),
                    "review_flags": event.get("review_flags", []),
                    "evidence": event.get("evidence", []),
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
                "conversation_id": session_id,
                "model": model_used,
                "usage": result.usage,
            },
        )
        await _save_memory_turn(request, session_id, trace_id, rewritten.answer)

        _tracer.end_span(
            span,
            result={
                "mode": "stream",
                "conversation_id": session_id,
                "model": model_used,
                "tool_events": len(result.tool_events),
                "sources": len(result.sources),
                "removed_fake_refs": rewritten.removed_fake_count,
                "memory_short_items": (
                    len(memory_context.short_term_messages) if memory_context else 0
                ),
                "memory_long_items": len(memory_context.long_term_items) if memory_context else 0,
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
