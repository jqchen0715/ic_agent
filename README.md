# IC-Expert Agent Showcase

面向集成电路问答与工程辅助的 Agent 项目。当前版本已经从“IC 知识问答服务”升级为：

**FastAPI + LangGraph + RAG + IC Tools + JSONL Memory + Optional Milvus + Web UI + Autonomous Agent + Reliability Audit**

## 核心能力

- IC 专业问答：支持 `/api/v1/chat` 非流式对话和 `/api/v1/chat/stream` SSE 流式对话。
- LangGraph 工具路由：`pre_tool_router -> tool_executor -> answer_generator`。
- IC RAG 检索：基于 LlamaIndex + Chroma，支持 dense + BM25/关键词 hybrid recall + reranker，返回 `source/page/chunk_id`。
- IC 范围分类：规则高置信 fast-path + LLM 结构化 fallback，统一决定是否进入知识库检索。
- 统一知识库构建：脚本构建、运行时自动建库、上传 PDF 增量入库共用 `KnowledgeBuilder`。
- 页码证据链：PDF 按页加载，chunk metadata 中保留 `page/page_start/page_end`。
- chunk 事实源一致：上传 PDF 后，数据库 `document_chunks` 与向量库 chunks 来自同一批入库对象。
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
- `answer_mode`：`strict_answer`、`assisted_draft` 或 `refusal`。
- `evidence_supported`：有工具证据支撑的结论。
- `draft_suggestions`：需要人工复核的草案建议。
- `missing_evidence`：缺失证据和可靠性边界。
- `next_actions`：建议继续补充或执行的动作。
- `reflection`：反思审查结果。
- `final_answer`：最终交付。

自主任务默认不是普通问答的替代品，而是 `Assisted Draft` 工作流：

```text
execute tools/reasoning
  -> audit_summary
  -> answer_mode
     -> strict_answer: 有工具证据、无复核风险，可以写成证据答案
     -> assisted_draft: 证据不足或存在复核标记，只能输出待复核草案
     -> refusal: 工具失败且无证据，拒绝交付实质结论
  -> final_answer 四段式输出
     -> 证据支持
     -> 草案建议
     -> 缺失证据
     -> 下一步
```

任务查询：

```text
GET /api/v1/agent/tasks
GET /api/v1/agent/tasks/{task_id}
```

### Document API

上传文档：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/upload \
  -F "file=@data/Verilog 硬件描述语言.pdf"
```

列出文档：

```bash
curl http://127.0.0.1:8000/api/v1/documents
```

上传 PDF 时会同步到 `DATA_PATH`，然后只增量更新这个 PDF 对应的 Chroma chunks，不会全量 rebuild。非 PDF 文件只写入数据库 chunks，不写入向量库。

## 知识库与 RAG 使用

### 目录与配置

默认使用：

```env
DATA_PATH=data
CHROMA_PATH=chroma_db
CHROMA_COLLECTION_NAME=ic_expert
EMBEDDING_MODEL_PATH=/path/to/bge-m3
EMBEDDING_DEVICE=cpu
SOURCE_MISMATCH_STRATEGY=rebuild
```

把 PDF 放入 `DATA_PATH` 后，可以通过脚本预构建知识库，也可以让第一次检索时自动构建。

### 脚本构建

```bash
uv run python scripts/build_knowledge.py \
  --data-dir data \
  --chroma-path chroma_db \
  --collection-name ic_expert \
  --embedding-model /path/to/bge-m3 \
  --embedding-device cpu
```

需要清空已有 collection/目录时：

```bash
uv run python scripts/build_knowledge.py --reset
```

脚本、上传 API、运行时自动建库都调用同一套 `app/core/rag/knowledge_builder.py`，避免不同入口切分策略、页码 metadata 或 chunk id 不一致。

### 上传增量入库

PDF 上传路径：

```text
UploadFile
  -> 保存到 uploads/
  -> 同步到 DATA_PATH
  -> KnowledgeBuilder.index_pdf()
  -> 删除该 PDF 旧 chunks
  -> 写入该 PDF 新 chunks 到 Chroma
  -> 用同一批 vector documents 写入 document_chunks
```

因此：

- 上传新 PDF 不会触发全量 rebuild。
- 同名文件会自动加 `doc_id` 前缀片段避免覆盖。
- `document_chunks.vector_id` 对齐向量库 chunk id。
- `document_chunks.meta` 保留 `source/file_name/file_path/file_hash/chunk_id/page/page_start/page_end/chunk_strategy`。

### Hybrid Retrieval

当前 `ic_rag_search` 的召回链路：

```text
query expansion
  -> dense retrieval from Chroma
  -> BM25/keyword retrieval from Chroma stored chunks
  -> merge by chunk_id/source/page
  -> rerank
  -> citation rewrite
```

BM25/关键词召回会保留代码和约束中的精确词，例如：

```text
fork
join
defparam
create_clock
set_input_delay
always_comb
```

可调配置：

```env
RAG_ENABLE_KEYWORD_RETRIEVAL=true
RAG_RETRIEVAL_CANDIDATE_K=20
RAG_KEYWORD_CANDIDATE_K=20
RAG_DENSE_WEIGHT=0.65
RAG_KEYWORD_WEIGHT=0.55
RAG_ENABLE_RERANKER=true
RAG_RERANK_TOP_K=10
RAG_RERANKER_MODEL=cross-encoder/bge-reranker-v2-m3
RAG_RERANKER_DEVICE=cpu
```

如果只想验证 dense 路径，可以临时设置：

```env
RAG_ENABLE_KEYWORD_RETRIEVAL=false
```

### IC Scope Classifier

普通问答和自主任务共用 `app/core/intent/domain_classifier.py`，避免两条链路各自维护不同的 IC 范围判断。

分类器遵循以下约束：

```text
用户问题
  -> 规则词表高置信命中 IC/Verilog
  -> 直接进入 RAG

规则不确定
  -> 调用 LLM 做结构化分类

LLM 只决定是否应该检索 IC 知识库
  -> 不允许直接生成答案

最终能不能回答
  -> 由 RAG 检索证据和 evidence gate 决定
```

LLM fallback 只输出结构化 JSON：

```json
{
  "in_scope": true,
  "confidence": "medium",
  "domain": "rtl_design",
  "normalized_query": "IC/Verilog RTL 乘法器结构设计方法",
  "reason": "问题涉及数字电路中的乘法器设计"
}
```

分类结果只影响工具路由和检索 query。例如：

```text
怎么设计一个乘法器
  -> rule / high / rtl_design
  -> rag_query = IC/Verilog RTL 怎么设计一个乘法器
  -> selected_tools = ["ic_rag_search"]
```

注意：`in_scope=true` 不代表可以直接回答。它只表示应该进入 IC 知识库检索；如果 RAG 没有检索到可引用证据，普通问答仍会严格拒答。

## Agent 主链路

普通问答链路：

```text
ChatRequest
  -> 读取短期/长期记忆
  -> LangGraphICAgent
  -> pre_tool_router
     -> ICDomainClassifier
     -> 规则命中或 LLM 分类后选择工具
  -> tool_executor
     -> ic_rag_search
  -> answer_generator
     -> evidence gate
  -> citation_rewriter
  -> 保存记忆
  -> ChatResponse / SSE events
```

强自主 Agent 链路：

```text
goal
  -> plan
  -> ICDomainClassifier 确保 IC 目标包含知识库检索步骤
  -> execute tools/reasoning
  -> collect evidence/confidence/review_flags
  -> recover on failure
  -> audit_summary
  -> answer_mode
  -> finalize as Evidence Answer / Assisted Draft / Refusal
  -> reflect
  -> remember final result
```

## 可靠性机制

项目不是简单让 LLM 自由发挥，而是通过工程约束提高可验证性：

- 工具调用前校验参数 schema，缺参数或多余参数会被拒绝。
- 工具输出统一归一化为审计记录。
- LLM scope classifier 只做结构化分类，不允许直接回答用户问题。
- RAG 未命中时普通问答严格拒答，自主任务进入 `assisted_draft` 或 `refusal`。
- 普通问答只输出有检索证据支撑的答案；自主任务可以生成草案，但必须用 `answer_mode` 和四段式交付标记证据边界。
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
- `app/core/intent/domain_classifier.py`：IC/Verilog 范围分类，规则 fast-path + LLM fallback。
- `app/core/memory`：JSONL 短期记忆、JSONL 长期关键词召回、可选 Milvus 长期记忆和记忆管理器。
- `app/core/rag`：统一知识库构建、dense/BM25 hybrid 检索、rerank 和引用治理。
- `app/core/tools`：工具注册、参数校验、审计归一化和 IC 工具实现。
- `app/static`：无构建前端页面、样式和交互逻辑。
- `app/etl`：文档解析与 IC 定制分块。
- `scripts/build_knowledge.py`：命令行构建 Chroma 知识库。
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
│   │   ├── intent/
│   │   ├── memory/
│   │   ├── rag/
│   │   └── tools/
│   ├── static/
│   ├── etl/
│   ├── infrastructure/
│   └── models/
├── evaluation/
├── scripts/
│   └── build_knowledge.py
├── data/
├── chroma_db/
├── tests/
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── README.pdf
└── README.md
```

## 快速开始

1. 创建虚拟环境并安装依赖。

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

如果已经误用了 `Python 3.11.0b*` 这类 beta 解释器创建 `.venv`，先重建环境：

```bash
uv run --python 3.12 python -V
```

确认输出是 `Python 3.12.x` 后，再运行脚本或服务。

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
- 不要使用 `Python 3.11.0b*` beta 解释器；Chroma/Numpy/SQLAlchemy 等依赖会在导入阶段失败。推荐 `Python 3.12.x`。
- RAG embedding 模型路径不可用时，`ic_rag_search` 会失败，自主 Agent 会记录工具错误并要求复核。
- 规则型 Verilog/SDC 工具适合做初筛，不替代 Verilator、Yosys、STA 工具或人工 signoff。
