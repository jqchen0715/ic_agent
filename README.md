# IC-Expert Agent Showcase

面向集成电路问答与工程辅助的最终展示仓库。

**核心定位：FastAPI + LangGraph + LlamaIndex + Chroma + IC Tools + Evaluation**

详细运行与排障手册见：`MERGED_USAGE_MANUAL.md`

## 展示目标

- 提供可直接演示的 IC 专业 Agent 服务（HTTP / SSE）。
- 通过 LangGraph 组织 Agent 推理与工具调用流程。
- 通过 LlamaIndex + Chroma 实现 IC 文档检索与可追溯引用。
- 内置 IC 专业工具（如 Verilog 检查、时序约束建议）。
- 提供评测集、评测脚本、评测报告，形成质量闭环。

## Agent 主链路

- `/api/v1/chat` 默认走 LangGraph 主链路：
  `pre_tool_router -> tool_executor -> answer_generator`。
- `/api/v1/agent/run` 提供强自主 Agent 模式：
  `plan -> execute tools/reasoning -> recover -> reflect -> finalize`。
- `pre_tool_router` 会按意图挑选 `ic_rag_search / verilog_code_analyzer / timing_constraint_suggester`。
- 对过短问题（如“乘法器”“时序”）先返回澄清问题，避免直接长答。
- `POST /api/v1/chat`（非流式）返回：`answer`、`trace_id`、`sources`、`tool_events`（兼容保留 `content`）。
- `POST /api/v1/chat/stream`（SSE）同样走 Agent 主链路，固定事件为：`tool_call`、`tool_result`、`answer`、`citation`、`done`、`error`。
- `POST /api/v1/agent/run` 返回自主任务的 `steps`、工具观察、失败恢复、`reflection` 与最终交付。
- 最终答案中的“参考资料”由服务端重写生成，仅允许本轮真实检索到的 `source/page`，伪引用会被移除并提示。
- 默认启用会话记忆：响应会返回 `conversation_id`，后续请求带上同一个 ID
  即可复用短期历史与长期召回记忆；本地记忆默认保存在 `data/memory`。
- 长期记忆可切换到 Milvus：设置 `MEMORY_BACKEND=milvus` 后会自动创建
  `agent_memory` 集合；若 Milvus 不可用，会自动退回本地 JSONL。

## 模块职责

- `app/api`：HTTP / SSE 接口。
- `app/core/agent`：LangGraph Agent 编排。
- `app/core/rag`：IC RAG 检索与引用处理。
- `app/core/tools`：IC 工具。
- `app/etl`：文档解析与 IC 分块。
- `evaluation`：评测集、评测脚本、报告。

## 目录结构

```text
project-python/
├── app/
│   ├── main.py                # FastAPI 应用入口
│   ├── config.py              # 全局配置
│   ├── api/                   # HTTP / SSE 路由层
│   ├── core/
│   │   ├── agent/             # LangGraph Agent 编排
│   │   ├── rag/               # IC RAG 检索与引用处理
│   │   └── tools/             # IC 工具
│   ├── etl/                   # 文档解析与 IC 分块
│   ├── infrastructure/        # 数据库、缓存、模型等基础设施封装
│   └── models/                # 数据模型定义
├── evaluation/
│   ├── datasets/              # 评测数据集
│   ├── reports/               # 评测报告输出
│   └── README.md              # 评测说明与约定
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 快速开始

1. 创建虚拟环境并安装依赖。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

2. 配置环境变量。

```bash
cp .env.example .env
```

3. 启动服务。

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. 健康检查。

```bash
curl http://127.0.0.1:8000/api/v1/health
```
