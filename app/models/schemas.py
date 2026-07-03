# -*- coding: utf-8 -*-
"""API 请求与响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import MessageRole


class ChatMessage(BaseModel):
    """单条对话消息。"""

    role: str = Field(description="角色：system | user | assistant")
    content: str = Field(description="文本内容")


class ChatRequest(BaseModel):
    """对话请求。"""

    messages: list[ChatMessage] = Field(min_length=1, description="OpenAI 风格消息列表")
    model: str | None = Field(default=None, description="优先使用的模型 ID")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    conversation_id: str | None = Field(default=None, description="可选会话 ID")


class ChatResponse(BaseModel):
    """非流式对话响应。"""

    id: str = Field(description="响应 ID")
    conversation_id: str | None = Field(
        default=None,
        description="会话 ID，用于后续请求延续记忆",
    )
    model: str = Field(description="实际使用的模型")
    answer: str = Field(description="助手回复正文（最终口径）")
    content: str | None = Field(default=None, description="兼容字段，等同 answer")
    trace_id: str | None = Field(default=None, description="链路追踪 ID")
    usage: dict[str, Any] | None = Field(default=None, description="Token 用量")
    sources: list[ICRetrievalResult] = Field(default_factory=list, description="答案引用来源")
    tool_events: list[dict[str, Any]] = Field(default_factory=list, description="工具调用事件列表")


class AutonomousTaskRequest(BaseModel):
    """自主 Agent 任务请求。"""

    goal: str = Field(min_length=1, description="用户希望 Agent 自主完成的目标")
    conversation_id: str | None = Field(default=None, description="会话 ID，用于任务记忆")
    model: str | None = Field(default=None, description="优先使用的模型 ID")
    max_steps: int = Field(default=6, ge=1, le=10, description="最多执行步骤数")


class AutonomousTaskStep(BaseModel):
    """自主 Agent 任务步骤。"""

    id: str
    title: str
    description: str = ""
    action_type: str = "reasoning"
    tool_name: str | None = None
    status: str = "pending"
    rationale: str | None = Field(default=None, description="执行该步骤/工具的原因")
    arguments: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list, description="工具返回的可审计证据")
    confidence: str = Field(default="unknown", description="high | medium | low | unknown")
    review_flags: list[str] = Field(default_factory=list, description="需要人工复核的原因")
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class AutonomousTask(BaseModel):
    """自主 Agent 任务结果。"""

    id: str
    session_id: str
    goal: str
    status: str
    created_at: str
    updated_at: str
    steps: list[AutonomousTaskStep] = Field(default_factory=list)
    answer_mode: str = Field(
        default="assisted_draft",
        description="strict_answer | assisted_draft | refusal",
    )
    evidence_supported: list[str] = Field(default_factory=list, description="有工具证据支撑的结论")
    draft_suggestions: list[str] = Field(default_factory=list, description="需人工复核的草案建议")
    missing_evidence: list[str] = Field(default_factory=list, description="缺失证据或可靠性边界")
    next_actions: list[str] = Field(default_factory=list, description="建议继续补充或执行的动作")
    final_answer: str | None = None
    reflection: dict[str, Any] = Field(default_factory=dict)
    audit_summary: dict[str, Any] = Field(default_factory=dict, description="任务级审计摘要")
    review_flags: list[str] = Field(default_factory=list, description="任务级人工复核原因")
    confidence: str = Field(default="unknown", description="任务级置信度")
    error: str | None = None


class AutonomousTaskListResponse(BaseModel):
    """自主 Agent 任务列表响应。"""

    tasks: list[AutonomousTask] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    """文档上传响应。"""

    document_id: str
    filename: str
    status: str
    chunk_count: int = 0
    message: str = "ok"


class DocumentInfo(BaseModel):
    """文档列表项。"""

    id: str
    filename: str
    mime_type: str | None = None
    status: str
    created_at: str | None = None


class DocumentUploadRequest(BaseModel):
    """文档上传附加元数据（可选）。"""

    tags: list[str] = Field(default_factory=list)


class Message(BaseModel):
    """对话消息（记忆/RAG 内部使用，含角色枚举）。"""

    role: MessageRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryItem(BaseModel):
    """长期记忆召回条目。"""

    id: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryContext(BaseModel):
    """短期 + 长期记忆合并上下文。"""

    session_id: str
    short_term_messages: list[Message]
    long_term_items: list[MemoryItem]


class RetrievalResult(BaseModel):
    """检索单条结果。"""

    id: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = "vector"


class ICRetrievalResult(BaseModel):
    """IC 检索标准输出（统一字段）。"""

    content: str
    source: str
    page: str = "页码未知"
    score: float = 0.0
    chunk_id: str


class Citation(BaseModel):
    """答案中的引用标注。"""

    index: int
    result_id: str
    snippet: str


class RAGResponse(BaseModel):
    """RAG 生成结果。"""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    raw_contexts: list[RetrievalResult] = Field(default_factory=list)
    model: str | None = None
