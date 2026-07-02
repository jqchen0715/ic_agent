# -*- coding: utf-8 -*-
"""强自主 Agent API。"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.api.routes.chat import _build_router
from app.core.agent.autonomous import AgentTaskStore, AutonomousAgent
from app.core.memory.factory import get_memory_manager
from app.core.tools.factory import build_ic_tool_registry
from app.models.schemas import (
    AutonomousTask,
    AutonomousTaskListResponse,
    AutonomousTaskRequest,
)

router = APIRouter(tags=["agent"])

_task_store = AgentTaskStore()


def _build_autonomous_agent() -> AutonomousAgent:
    return AutonomousAgent(
        model_router=_build_router(),
        tool_registry=build_ic_tool_registry(),
        memory_manager=get_memory_manager(),
    )


@router.post("/agent/run", response_model=AutonomousTask)
async def run_agent_task(request: AutonomousTaskRequest) -> AutonomousTask:
    """同步执行一个强自主 Agent 任务并返回完整执行轨迹。"""
    session_id = request.conversation_id or str(uuid.uuid4())
    try:
        agent = _build_autonomous_agent()
        task = await agent.run(
            goal=request.goal,
            session_id=session_id,
            model_preference=request.model,
            max_steps=request.max_steps,
        )
        _task_store.save(task)
        return task
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("自主 Agent 任务失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"自主 Agent 任务失败: {exc!s}") from exc


@router.get("/agent/tasks", response_model=AutonomousTaskListResponse)
async def list_agent_tasks(limit: int = 20) -> AutonomousTaskListResponse:
    """查看最近的自主 Agent 任务。"""
    return AutonomousTaskListResponse(tasks=_task_store.list_recent(limit=limit))


@router.get("/agent/tasks/{task_id}", response_model=AutonomousTask)
async def get_agent_task(task_id: str) -> AutonomousTask:
    """按 ID 查看自主 Agent 任务。"""
    task = _task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task
