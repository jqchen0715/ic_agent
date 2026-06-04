``` python
# -*- coding: utf-8 -*-
"""对话 API：非流式与流式输出。"""

from __future__ import annotations

import json
#：生成全局唯一 trace_id 链路追踪 ID。
import uuid
#Any, AsyncIterator：类型注解，AsyncIterator 用于异步流式生成器。
from typing import Any, AsyncIterator
'''APIRouter：接口路由拆分，模块化管理 /chat、/chat/stream 接口。
HTTPException：主动抛出 HTTP 标准异常。
StreamingResponse：实现SSE 流式响应，适配对话打字机效果。
'''
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from loguru import logger
#读取项目全局配置：模型 Key、BaseURL、默认模型名等。
from app.config import get_settings
'''LangGraphICAgent：基于 LangGraph 实现的智能 Agent 主链路（路由判断、工具调用、多轮编排）。
rewrite_answer_citations：RAG 引用改写，清洗虚假引用、格式化文献标注。
build_ic_tool_registry：工具工厂，注册业务工具集，供 Agent 调用。'''
from app.core.agent.langgraph_agent import LangGraphICAgent
from app.core.rag.citation_rewriter import rewrite_answer_citations
from app.core.tools.factory import build_ic_tool_registry

from app.infrastructure.llm.model_router import ModelConfig, ModelRouter
'''Tracer：链路追踪组件，记录接口全流程耗时、调用节点、异常信息。'''
from app.infrastructure.trace.tracer import Tracer
'''schema 校验：检查用户发来的请求格式是否正确，比如有没有 messages、字段类型对不对；不对就直接报错，避免后面出错'''
from app.models.schemas import ChatRequest, ChatResponse
'''新建一个 独立子路由，用来统一管理当前文件下所有对话接口：
/chat 非流式接口
/chat/stream 流式接口
实现接口模块化拆分，不用把所有接口都堆在项目主入口文件。
参数 tags=["chat"]
给这一组接口打上分类标签；
在 FastAPI 自动接口文档（Swagger / Redoc）里，会把这两个接口归类到 chat 分组，方便查看和调试。
'''
router = APIRouter(tags=["chat"])
#做一个追踪
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
        '''这两个参数是为后续多模型接入、自动降级、负载均衡预留的设计。
'''
        priority=0,#模型优先级，数值越小优先使用；当多个模型可用时，优先级高的会被选中。这里设置为0，表示最高优先级。
        weight=1.0,#模型权重，用于负载均衡；当多个模型优先级相同且可用时，权重高的被选中的概率更大。这里设置为1.0，表示默认权重。
    )
    return ModelRouter([cfg])


def _build_ic_agent() -> LangGraphICAgent:
    """构造 IC LangGraph 主链路 Agent。"""
    return LangGraphICAgent(
        model_router=_build_router(),
        tool_registry=build_ic_tool_registry(),
    )

‘’‘.post 声明接口只接收 POST 请求
为什么对话接口用 POST？
要传大量对话消息（messages），不适合放 URL 路径参数
支持复杂 JSON 请求体
更安全，不会把对话内容暴露在地址栏’‘’
@router.post("/chat", response_model=ChatResponse)#强制约束返回格式接口 return 出来的数据必须和 ChatResponse 的字段一致
async def chat(request: ChatRequest) -> ChatResponse:
    """非流式对话：走 LangGraph 主链路（pre_tool_router -> tools -> final answer）。"""
    trace_id = str(uuid.uuid4())
    span = _tracer.start_trace(trace_id, "chat")

    try:
        #model_dump()：把 Python 对象转成 dict
        messages = [m.model_dump() for m in request.messages]
        #调用agent这个方法
        agent = _build_ic_agent()
        #这里需要等待一个耗时操作
        result = await agent.run(
            messages=messages,
            model_preference=request.model,
            #温度控制回答随机性
            temperature=request.temperature,
            #模型最多生成多少 token作为回答，避免生成过长的回答导致资源浪费或响应过慢
            max_tokens=request.max_tokens,
        )
        #项目做了“引用后处理”校验引用
        #删除假引用
        #重写 citation
        rewritten = rewrite_answer_citations(result.content, result.sources)
        #项目专门处理了 hallucination （幻觉）/ fake citation（虚假引用）
        model_used = result.model_id or request.model or get_settings().openai_model
        logger.info(
            "LangGraph 路由完成 trace={} tools={} reason={} clarify={} removed_fake_refs={}",
            trace_id,
            result.selected_tools,
            #为什么选择这些工具
            result.route_reason,
            #是否需要用户补充信息
            result.needs_clarification,
            #删除了多少个虚假引用
            rewritten.removed_fake_count,
        )
        #一次请求结束后，把整个运行过程的信息写入 tracing 系统
        _tracer.end_span(
            #一次聊天的生命请求
            span,
            result={
                "model": model_used,
                #token 消耗统计，哪个请求最烧 token？
                #哪个工具导致上下文变大？
                #哪个用户最耗费？
                "usage": result.usage,
                "tools": result.selected_tools,
                "tool_events": len(result.tool_events),
                "sources": len(result.sources),
                "route_reason": result.route_reason,
                #支持澄清问题
                "needs_clarification": result.needs_clarification,
                "removed_fake_refs": rewritten.removed_fake_count,
            },
        )
‘’‘搞tracing就是为了弄清楚哪里出问题，而id就是为了方便做追踪请求’‘’

        return ChatResponse(                #FastAPI 最终返回一个结构化响应
            id=str(uuid.uuid4()),
            model=model_used,
            ‘’‘answer：更符合问答接口语义，表示“模型最终答案”。
            content：更通用，方便前端或历史接口按消息内容读取。
            两个字段都保留，是为了兼容不同调用方，避免有的地方读 response.answer，有的地方读 response.content 时出错。
            ’‘’
            answer=rewritten.answer,
            content=rewritten.answer,
            #id返回给前端，方便前端做请求追踪和日志关联；前端日志里有这个 id，就能知道这是哪个请求的响应了。
            trace_id=trace_id,
            usage=result.usage,
            sources=result.sources,
            #前端展示工具调用信息，帮助用户理解模型是怎么得出这个答案的
            tool_events=result.tool_events,
        )
        #HTTPException 是 Web 开发框架（尤其是 FastAPI 和 Starlette）中提供的一个特殊异常类。
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chat 失败: {}", exc)
        _tracer.end_span(span, error=str(exc))
        raise HTTPException(status_code=500, detail=f"对话失败: {exc!s}") from exc   #保留原始异常链


async def _stream_generator(
    request: ChatRequest,
    trace_id: str,
    span: object,
) -> AsyncIterator[bytes]:    #Iterator，“可以不断 yield 数据”
    """SSE 风格流：每行 data: {json}\\n\\n。"""#Server-Sent Events
    def _emit(event: str, payload: dict[str, Any]) -> bytes:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
        #ensure_ascii=False，为了防止把中文转 unicode

#把工具返回结果整理成适合展示/日志记录的短文本（压缩输出结果）

    def _summarize_tool_result(raw: Any, max_len: int = 240) -> str:
        #这里的 raw 通常是指从数据库、API 接口或者用户输入中直接拿到的原始值，strip 去掉首尾空格
        ‘’‘去除多余换行
            去除多余空格
            让日志更干净’‘’
        text = " ".join(str(raw or "").strip().split())
        #短文本直接返回，长文本截断只保留240个字符，后面加省略号，避免日志被工具结果撑爆了
        if not text:
            return "（空结果）"
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    try:
        #输入层，数据标准化
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
        #Agent 会记录工具调用事件
        for event in result.tool_events:
            tool_name = str(event.get("tool", "unknown_tool"))
            tool_args = event.get("arguments", {})
            #arguments 可能格式错，必须校验类型，前端可能炸
            if not isinstance(tool_args, dict):
                tool_args = {}
            #正在推送“工具调用事件”
            yield _emit(
                "tool_call",
                {
                    "trace_id": trace_id,
                    "tool": tool_name,
                    #里面包含了执行该工具所需的具体信息，比如工具名称、参数等；前端可以根据这些信息展示工具调用的过程，或者做一些特殊处理（比如对特定工具调用结果进行高亮展示）。
                    "arguments": tool_args,
                },
            )

            yield _emit(
                "tool_result",
                {
                    "trace_id": trace_id,
                    "tool": tool_name,
                    #使用布尔值判断是否调用成功？
                    "ok": bool(event.get("ok", False)),
                    #不直接返回tool结果，而是摘要后再输出
                    "summary": _summarize_tool_result(event.get("result", "")),
                },
            )
        #真正把模型答案一段段推给前端
        content = rewritten.answer or ""
        chunk_size = 64     #每次发送 64 个字符
        #range(start, end, step)，从 start 到 end，每次加 step
        for idx in range(0, len(content), chunk_size):
            #取一小段字符串，比如第一次：content[0:64]
            delta = content[idx : idx + chunk_size] #delta指的是新增字符


            #如果这一块不是空字符串
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

#定义一个流式聊天接口
@router.post("/chat/stream")#前端调用 /chat/stream，当用户 POST 请求 /chat/stream，执行chat_stream()
#get不适合传这种复杂的json，所有用post
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """流式输出（Server-Sent Events 兼容格式）。"""
    trace_id = str(uuid.uuid4())
    span = _tracer.start_trace(trace_id, "chat_stream")

    return StreamingResponse(
        _stream_generator(request, trace_id, span),
        media_type="text/event-stream",#这是 SSE 协议,前端只认这个格式的响应，才能正确处理流式数据
        headers={
            "Cache-Control": "no-cache",#禁止缓存 streaming，streaming 是实时数据
            "Connection": "keep-alive",#不要断开 HTTP 连接，因为：SSE 必须长连接
            "X-Trace-Id": trace_id,
        },
    )

