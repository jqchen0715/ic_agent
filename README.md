# IC-Expert Agent Showcase

面向集成电路问答与工程辅助的 Agent 项目。当前版本已经从“IC 知识问答服务”升级为：

**FastAPI + LangGraph + RAG + IC Tools + JSONL Memory + Optional Milvus + Web UI + Autonomous Agent + Reliability Audit**

详细运行与排障手册见：`MERGED_USAGE_MANUAL.md`

项目讲解与面试掌握手册见：`PROJECT_MASTERY_MANUAL.md`

## 核心能力

- IC 专业问答：支持 `/api/v1/chat` 非流式对话和 `/api/v1/chat/stream` SSE 流式对话。
- LangGraph 工具路由：`pre_tool_router -> tool_executor -> answer_generator`。
- IC RAG 检索：基于 LlamaIndex + Chroma + embedding + reranker，返回 `source/page/chunk_id`。
- 服务端引用治理：最终引用只允许来自本轮真实检索结果，移除模型自造引用。
- IC 工具调用：内置 `ic_rag_search`、`verilog_code_analyzer`、`timing_constraint_suggester`。
- 记忆系统：默认短期 JSONL 历史窗口 + 长期 JSONL 关键词召回，支持 `conversation_id` 多轮上下文复用。
- 可选 Milvus 长期记忆：设置 `MEMORY_BACKEND=milvus` 后，仅将长期记忆替换为 Milvus 向量召回。
- 强自主 Agent：`/api/v1/agent/run` 支持计划、执行、失败恢复、反思、审计和最终交付。
- 可靠性审计：工具结果统一带 `evidence`、`confidence`、`review_flags`、`summary`。
- Web 前端：访问 `/` 即可使用聊天、工具/来源展示、会话记忆和自主任务面板。
- 质量闭环：保留 RAGAS、自定义引用正确率、拒答正确率和工具路由准确率评测脚本。

## 入口与接口

### 前端

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

打开：

```text
http://127.0.0.1:8000/
```

如果 8000 被占用，可以改用其他端口，例如 `8123`。

### Chat API

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "setup violation 怎么优化？"}
    ]
  }'
```

响应包含：

- `answer`：最终回答。
- `conversation_id`：会话 ID，后续请求带上后可复用记忆。
- `sources`：真实检索来源。
- `tool_events`：工具调用审计事件。

### Stream API

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Verilog 里 blocking 和 nonblocking 有什么区别？"}
    ]
  }'
```

SSE 事件包括：

- `tool_call`
- `tool_result`
- `answer`
- `citation`
- `done`
- `error`

### Autonomous Agent API

```bash
curl -X POST http://127.0.0.1:8000/api/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "分析这段 Verilog 是否有 latch 风险，并给出可复核建议",
    "max_steps": 6
  }'
```

返回内容包括：

- `steps`：计划和执行轨迹。
- `rationale`：每一步为什么执行。
- `evidence`：工具证据。
- `confidence`：任务和步骤置信度。
- `review_flags`：需要人工复核的原因。
- `audit_summary`：任务级审计摘要。
- `reflection`：反思审查结果。
- `final_answer`：最终交付。

任务查询：

```text
GET /api/v1/agent/tasks
GET /api/v1/agent/tasks/{task_id}
```

## Agent 主链路

普通问答链路：

```text
ChatRequest
  -> 读取短期/长期记忆
  -> LangGraphICAgent
  -> pre_tool_router
  -> tool_executor
  -> answer_generator
  -> citation_rewriter
  -> 保存记忆
  -> ChatResponse / SSE events
```

强自主 Agent 链路：

```text
goal
  -> plan
  -> execute tools/reasoning
  -> collect evidence/confidence/review_flags
  -> recover on failure
  -> finalize
  -> reflect
  -> audit_summary
  -> remember final result
```

## 可靠性机制

项目不是简单让 LLM 自由发挥，而是通过工程约束提高可验证性：

- 工具调用前校验参数 schema，缺参数或多余参数会被拒绝。
- 工具输出统一归一化为审计记录。
- RAG 未命中时严格拒答或标记 `needs_review`。
- SDC 模板会标记默认假设，例如模块名、时钟周期、IO delay 缺失。
- Verilog 工具会返回规则命中的证据和风险标记。
- 自主 Agent 遇到工具失败会降级恢复，但任务状态会变成 `needs_review`，不会伪装成完成。
- 反思模型不可用时会降级为保守评分。

## 记忆系统

默认本地记忆：

```env
MEMORY_ENABLED=true
MEMORY_BACKEND=local
MEMORY_STORE_PATH=data/memory
MEMORY_WINDOW_SIZE=20
MEMORY_RECALL_TOP_K=5
```

默认情况下：

- `ShortTermMemory`：使用 JSONL 保存最近多轮 user/assistant 消息。
- `LongTermMemory`：使用 JSONL 保存长期条目，并通过简单关键词重叠召回。
- `MEMORY_BACKEND=milvus` 只把长期记忆替换成 Milvus 向量召回，短期记忆仍保持 JSONL。

Milvus 长期记忆：

```env
MEMORY_BACKEND=milvus
MEMORY_MILVUS_COLLECTION_NAME=agent_memory
MEMORY_EMBEDDING_MODEL_PATH=BAAI/bge-m3
MEMORY_EMBEDDING_DEVICE=cpu
```

Milvus 不可用时，长期记忆会回退到本地 JSONL 关键词召回。

## 模块职责

- `app/main.py`：FastAPI 应用入口，挂载 API 和静态前端。
- `app/api/routes/chat.py`：普通对话和 SSE 对话入口，负责记忆读写、Agent 调用和引用重写。
- `app/api/routes/agent.py`：强自主 Agent 任务入口。
- `app/core/agent/langgraph_agent.py`：普通问答 Agent 状态图。
- `app/core/agent/autonomous.py`：强自主 Agent 计划、执行、恢复、反思和审计。
- `app/core/memory`：JSONL 短期记忆、JSONL 长期关键词召回、可选 Milvus 长期记忆和记忆管理器。
- `app/core/rag`：IC RAG 检索、rerank 和引用治理。
- `app/core/tools`：工具注册、参数校验、审计归一化和 IC 工具实现。
- `app/static`：无构建前端页面、样式和交互逻辑。
- `app/etl`：文档解析与 IC 定制分块。
- `evaluation`：评测集、评测脚本和报告。

## 目录结构

```text
project-python/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── api/routes/
│   │   ├── chat.py
│   │   ├── agent.py
│   │   ├── document.py
│   │   └── health.py
│   ├── core/
│   │   ├── agent/
│   │   ├── memory/
│   │   ├── rag/
│   │   └── tools/
│   ├── static/
│   ├── etl/
│   ├── infrastructure/
│   └── models/
├── evaluation/
├── tests/
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── PROJECT_MASTERY_MANUAL.md
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

## 当前注意事项

- LLM API 不可用时，自主 Agent 会走 fallback，并将结果标记为 `needs_review`。
- RAG embedding 模型路径不可用时，`ic_rag_search` 会失败，自主 Agent 会记录工具错误并要求复核。
- 规则型 Verilog/SDC 工具适合做初筛，不替代 Verilator、Yosys、STA 工具或人工 signoff。
