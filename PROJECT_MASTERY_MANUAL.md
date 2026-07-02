# IC-Expert Agent 项目掌握手册
这个项目的主线可以这样记：

  普通问答链路：

  /api/v1/chat 或 /api/v1/chat/stream -> 读取会话记忆 -> 构造 LangGraphICAgent ->
  pre_tool_router 选工具 -> tool_executor 调 IC 工具/RAG -> 工具审计归一化 ->
  answer_generator 基于工具结果生成回答 -> rewrite_answer_citations 服务端重写引用 ->
  保存会话记忆 -> 返回 ChatResponse / SSE events。

  强自主 Agent 链路：

  /api/v1/agent/run -> AutonomousAgent -> plan 任务拆解 -> execute 工具/推理步骤 ->
  收集 evidence/confidence/review_flags -> 工具失败时 recover -> finalize 最终交付 ->
  reflect 反思审查 -> audit_summary 汇总 -> 保存任务记忆。

  当前版本的关键词：

  - 前端：`app/static` 提供可直接打开的 Web UI，包含聊天、工具/来源、自主任务面板。
  - 记忆：`app/core/memory` 默认提供 JSONL 短期历史 + JSONL 长期关键词召回，Milvus 只作为长期记忆可选增强。
  - 自主：`app/core/agent/autonomous.py` 支持计划、执行、恢复、反思和审计。
  - 可靠：工具调用有参数校验，工具结果统一输出 evidence/confidence/review_flags。

  P0 必看

  1. app/main.py:15
     掌握 FastAPI 启动生命周期、数据库 engine 初始化、API 路由挂载、静态前端挂载。
  2. app/config.py:9
     掌握所有运行配置：模型、数据库、Chroma、embedding、reranker、data 路径、JSONL 记忆路径、可选 Milvus 长期记忆配置。
  3. app/api/routes/chat.py:50
     普通问答最核心入口。必须讲清楚非流式 /chat 和 SSE /chat/stream 怎么读取记忆、构造模型路由器、工具注册表、
     Agent、trace、引用重写，并在结束时保存记忆。
  4. app/core/agent/langgraph_agent.py:70
     普通问答 Agent 的最重要文件。重点掌握：
     _build_graph、_pre_tool_router、_tool_executor、_answer_generator、严格拒答、source 提取。
     这是你面试里讲“Agent 编排”的主战场。
  5. app/core/agent/autonomous.py
     强自主 Agent 主文件。重点掌握：
     _plan、_heuristic_plan、_invoke_tool_with_audit、_recover、_finalize、_reflect、_build_audit_summary。
     这是你讲“从问答机器人升级为强自主 Agent”的主战场。
  6. app/core/tools/base.py、app/core/tools/registry.py、app/core/tools/builtin/ic_tools.py
     掌握工具 schema 校验、invoke_with_audit 审计归一化，以及三个 IC 工具：
     ic_rag_search、verilog_code_analyzer、timing_constraint_suggester。
  7. app/core/memory/factory.py、app/core/memory/short_term.py、app/core/memory/long_term.py、app/core/memory/manager.py
     掌握 JSONL 短期历史、JSONL 长期关键词召回，以及 Milvus 作为可选长期记忆增强的切换逻辑。
  8. app/static/index.html、app/static/app.js、app/static/styles.css
     掌握前端如何调用流式 chat、自主任务 API，以及如何展示工具调用、来源、置信度和复核标记。
  9. app/core/rag/retriever.py:97
     掌握 LlamaIndex + Chroma + HuggingFace embedding + source 一致性检查 + reranker 的检索链路。重点看
     _ensure_index 和 retrieve。
  10. app/core/rag/citation_rewriter.py:31
     掌握“服务端引用治理”：移除模型自己编的参考资料，只保留本轮真实检索到的 source/page。这是项目亮点。
  11. app/api/routes/document.py:56
     掌握文档上传：保存文件、ETL 分块、写数据库、PDF 同步到 data/、重建 Chroma。

  P1 需要掌握

  1. app/etl/parser.py:27、app/etl/chunker.py:28、app/etl/ic_text_splitter.py:13
     掌握 PDF/TXT 解析、固定/递归/段落/IC 定制分块，尤其是保留 Verilog 代码块和章节边界。
  2. app/infrastructure/llm/model_router.py:41、app/infrastructure/llm/circuit_breaker.py:24
     掌握模型路由、优先级、加权选择、失败降级、熔断器。
  3. app/models/schemas.py:20
     掌握 API 请求响应结构，尤其 ChatRequest、ChatResponse、AutonomousTask、AutonomousTaskStep、ICRetrievalResult。
  4. app/infrastructure/database/models.py:23、app/infrastructure/database/session.py:36
     掌握 ORM 模型和异步 session。当前聊天主链路没有把对话消息落库，但文档上传会写 documents / document_chunks。
  5. evaluation/evaluate_ragas.py:238、evaluation/retrieval_smoke_test.py:85
     掌握项目质量闭环：RAGAS、引用正确率、拒答正确率、工具路由准确率、检索冒烟测试。
  6. tests/test_memory.py、tests/test_tool_audit.py、tests/test_autonomous_agent_audit.py
     掌握记忆、工具审计和自主 Agent 复核状态的回归测试思路。

  P2 了解即可
  app/core/agent/react_agent.py、planner.py、orchestrator.py、cache/redis_cache.py、
  vectordb/milvus_client.py。这些是通用 Agent/缓存/Milvus 能力储备，代码完整，但当前普通问答主链路主要走
  LangGraphICAgent，强自主任务主链路走 AutonomousAgent。

  学习顺序建议：

  1. 先读 README.md 和 app/main.py，理解入口和路由挂载。
  2. 再读 chat.py -> langgraph_agent.py，掌握普通问答主链路。
  3. 然后读 tools/base.py -> tools/registry.py -> ic_tools.py，掌握工具校验和审计。
  4. 接着读 autonomous.py -> agent.py，掌握强自主 Agent。
  5. 再读 memory/*，掌握短期 JSONL、长期 JSONL 关键词召回和可选 Milvus 长期记忆。
  6. 最后补 retriever.py -> citation_rewriter.py -> document.py -> etl/* -> evaluation/*。
> 目标：把这个项目从“能跑起来”升级为“能讲清楚、能改代码、能应对实习面试追问”的工程资产。

本项目是一个面向集成电路（IC）领域的专业 AI Agent 服务，核心能力是：用户通过 HTTP、SSE 或 Web 前端发起 IC 问答/自主任务请求，系统通过 LangGraph 和 AutonomousAgent 编排工具调用，结合 RAG 检索、Verilog 代码分析、时序约束建议、JSONL 会话记忆、可选 Milvus 长期记忆、工具审计和服务端引用治理，输出可追溯、可复核、低幻觉的专业回答或任务交付。

---

## 第一阶段：项目全局拆解（建立认知框架）

### 1. 项目解决的核心问题是什么？

一句话：

> 这个项目解决的是“IC 专业知识问答不可靠、缺少引用、不能调用领域工具、不能持续记忆上下文、不能自主拆解任务”的问题，通过 Agent + RAG + IC 工具链 + 记忆系统 + 自主任务执行，把普通大模型封装成一个面向集成电路学习和工程辅助的专业 Agent 系统。

业务视角：

- 对学生：可以用它查询 Verilog、时序分析、乘法器优化、ASIC/FPGA 等 IC 知识，并看到资料来源。
- 对工程场景：可以辅助检查 Verilog 代码、生成 SDC 时序约束建议、检索内部 IC 文档。
- 对连续学习：可以通过 `conversation_id` 复用短期上下文和长期记忆，避免每轮都从零开始。
- 对自主任务：可以输入一个目标，让 Agent 自动拆计划、调用工具、失败恢复、反思审查并给出最终交付。
- 对面试展示：它不是一个简单 ChatGPT wrapper，而是一个具备领域路由、工具调用、检索增强、记忆、自主执行、引用治理、可靠性审计和评测闭环的垂类 Agent 系统。

面试 1 分钟表达：

> 我做的是一个面向集成电路领域的 AI Agent 服务。用户通过 FastAPI 的 `/chat` 或 `/chat/stream` 提问后，系统会先读取会话记忆，再进入 LangGraph 主链路，由 `pre_tool_router` 判断问题类型，比如普通 IC 知识、Verilog 代码、时序约束，然后调用对应工具。知识类问题会走 LlamaIndex + Chroma 的 RAG 检索，Verilog 问题可以走规则型代码分析，时序问题可以生成 SDC 约束建议。最终回答由大模型基于工具结果生成，并且服务端会重写引用，只允许展示本轮真实检索到的 source/page。除此之外，我还加了 `/agent/run` 强自主 Agent，可以对目标自动规划、执行工具、失败恢复、反思审查，并把 evidence、confidence、review_flags 返回给前端。项目还有 JSONL 短期历史、JSONL 长期关键词召回、可选 Milvus 长期记忆、Web UI、ETL 建库、IC 定制分块、CrossEncoder 重排和 RAGAS 评测脚本，形成从数据、检索、生成、自主执行到评测的完整闭环。
``` text
etl建库:ETL 建库就是：把原始文档变成可以被 RAG 检索的知识库。
ETL 是三个词：
  - Extract：抽取
    从 PDF/TXT 里抽取文本。
    项目里对应 app/etl/parser.py。
  - Transform：转换
    清洗文本、切分 chunk、保留 Verilog 代码块/章节结构。
    项目里对应 app/etl/chunker.py 和 app/etl/ic_text_splitter.py。
  - Load：加载
    把切好的文本块写入数据库或向量库，后续用户提问时可以检索。
    项目里一部分写 PostgreSQL 的 documents/document_chunks，一部分写 Chroma 向量库。

  在这个项目里，ETL 建库流程大概是：

  PDF 文档
    -> 解析出文本
    -> 按 IC 语义分块
    -> embedding 向量化
    -> 写入 Chroma
    -> 用户提问时做相似度检索

  举例：你上传一本 Verilog 硬件描述语言.pdf，系统不会直接把整本书丢给大模型，而是先把 PDF 解析成文本，再切成很多小片段，比如“fork 语句是否可综合”“注释语法”“parameter 规则”等。每个片段会生成向量写入 Chroma。用户问“fork 可以综合吗？”，系统就从 Chroma 找到最相关片段，再让模型基于片段回答。
  所以简历里说「ETL 建库」，意思就是你做了从原始资料到可检索向量知识库的完整数据处理链路。
```
```
CrossEncoder 重排就是：先用向量检索粗召回一批候选文档，再用更精细的模型重新排序，把最相关的片段排到前面。

  在 RAG 里通常分两步：

  用户问题
    -> 向量检索召回 top 20
    -> CrossEncoder 对 query + 每个文档片段逐对打分
    -> 重新排序
    -> 取 top 3 / top 5 给大模型

  为什么需要它？

  向量检索一般是 Bi-Encoder 模式：

  query -> 向量
  document -> 向量
  两边向量做相似度计算

  优点是快，适合从大量文档里召回候选。缺点是它提前把 query 和 document 分开编码，细节匹配能力没那么强。

  CrossEncoder 不一样：

  [用户问题, 文档片段] -> 同一个模型一起读 -> 输出相关性分数

  它会同时看到问题和文档内容，所以判断更准。缺点是慢，因为每个候选片段都要跑一次模型，所以一般只用于“重排候选结果”，不直接全库搜索。

  在你这个项目里，对应代码是：

  - app/core/rag/retriever.py:308 里先检索候选，再调用 _run_reranker
  - app/core/rag/reranker.py:12 里用 sentence_transformers.CrossEncoder

  核心逻辑是：

  pairs = [(query, document.content) for document in documents]
  scores = model.predict(pairs)

  然后按 scores 从高到低排序。

  举例：

  用户问：

  fork 语句是否可以综合？

  向量检索可能召回：

  1. fork/join 并行语句
  2. Verilog initial 语句
  3. 综合与仿真的区别
  4. task/function 说明

  CrossEncoder 会逐对判断：

  ("fork 语句是否可以综合？", "fork 语句不可综合") -> 高分
  ("fork 语句是否可以综合？", "initial 用于仿真") -> 低分

  最后把“fork 语句不可综合”排到最前面。

  一句话面试版：

  > 我在 RAG 里使用 CrossEncoder 做二阶段重排。第一阶段用 bge-m3 向量检索快速召回候选片段，第二阶段把 query 和每个候选 chunk 拼成 pair 输入 CrossEncoder，重新
  > 计算相关性分数，再取 top-k 给 LLM。这样比单纯向量相似度更能捕捉细粒度语义匹配，能提升最终上下文质量。

```
```
RAGAS 是一个专门评估 RAG 系统效果的开源评测框架。

  它不是用来做检索或生成的，而是用来回答这个问题：

  > 我的 RAG 系统回答得准不准？检索上下文有没有用？回答有没有忠实于资料？有没有幻觉？

  在你这个项目里，RAGAS 评测脚本是 evaluation/evaluate_ragas.py。

  它会读取评测集：

  evaluation/datasets/eval_dataset_qg_30.json

  然后逐条调用最终 /chat 主链路，也就是：

  问题 -> LangGraph Agent -> RAG 检索 -> 生成答案 -> 引用重写

  再计算指标。

  项目里用了这些 RAGAS 指标：

  - faithfulness：忠实度
    判断答案里的内容是否能被检索上下文支持，越高说明越不容易幻觉。
  - answer_relevancy：答案相关性
    判断回答是否切中用户问题。
  - context_recall：上下文召回率
    判断检索到的上下文是否覆盖标准答案需要的信息。
  - context_precision：上下文精确率
    判断检索到的上下文里有多少是真正有用的，避免召回一堆噪声。

  你项目还加了几个自定义指标：

  - citation_correctness：引用正确率
    检查答案里的引用来源是否来自本轮真实检索结果。
  - refusal_correctness：拒答正确率
    对不该回答或检索不到的问题，检查系统是否正确严格拒答。
  - tool_routing_accuracy：工具路由准确率
    检查 Agent 是否选对工具，比如知识问答是否走 ic_rag_search，时序问题是否走 timing_constraint_suggester。

```
```
pre_tool_router 是这个项目里 LangGraph 主链路的“前置路由器”，负责在真正调用工具前做意图判断与工具选择。它的职责是：判空/判短问题要不要澄清、是否属于 IC 领域、以及需要调用哪些工具（RAG、Verilog 分析、时序约束）。在代码里它就是 langgraph_agent.py:108 的 LangGraphICAgent._pre_tool_router()。
它具体做了这些判断（精简版）：
空问题或过短问题 → 进入澄清（needs_clarification=True）。
非 IC 领域 → 不调用工具，后续走严格拒答。
IC 知识类 → 选择 ic_rag_search。
Verilog + 有代码 → 选 ic_rag_search + verilog_code_analyzer。
时序相关 → 选 ic_rag_search + timing_constraint_suggester。

澄清”在这里是指：当用户的问题太短、太泛或信息不足时，系统先不直接回答，而是要求用户补充上下文，再决定调用哪些工具并给出更精准的回答。

在这个项目里，澄清触发的典型情况是：
用户问题为空。
问题过短（比如“乘法器”“时序”这种词）。
```
---

### 2. 技术栈选择的原因

这个项目的技术栈不是简单堆库，而是围绕“垂类 Agent 服务”拆成几层。

#### FastAPI

为什么用 FastAPI：

- 适合构建 Python AI 服务，异步能力好，方便接 LLM 调用、检索和 SSE 流式输出。

- 自动生成 OpenAPI 文档，适合项目展示和联调。
- 请求/响应模型可以和 Pydantic 结合，接口结构清晰。
```
 SSE 全称是 Server-Sent Events，意思是：服务端持续向客户端推送事件流。

  在普通 HTTP 接口里，请求和响应是一次性的：

  客户端发请求 -> 服务端处理完 -> 一次性返回完整结果

  SSE 是：

  客户端发请求 -> 服务端保持连接 -> 服务端不断推送数据块 -> 最后结束

  它很适合做大模型流式输出，比如“打字机效果”。

  在你这个项目里，SSE 接口是：

  POST /api/v1/chat/stream

  对应代码在 app/api/routes/chat.py:216。

  返回类型是：

  StreamingResponse(..., media_type="text/event-stream")

  项目里定义了这些事件：

  tool_call     工具开始调用
  tool_result   工具调用结果
  answer        答案分片
  citation      引用来源
  done          结束
  error         异常

  SSE 返回的数据格式大概是：

  event: answer
  data: {"trace_id": "...", "chunk": "乘法器时序优化可以从"}

  event: answer
  data: {"trace_id": "...", "chunk": "流水线、结构优化、约束检查等方面入手"}

  event: done
  data: {"trace_id": "...", "model": "qwen-turbo-2025-02-11"}

  和 WebSocket 的区别：

  - SSE 是单向的：服务端 -> 客户端
  - WebSocket 是双向的：客户端 <-> 服务端
  - SSE 基于 HTTP，实现简单，适合模型流式回答
  - WebSocket 更适合实时协作、游戏、双向通信

  面试可以这样说：

  > SSE 是一种基于 HTTP 的服务端事件推送机制。我在项目中用 StreamingResponse 实现 /chat/stream，把 Agent 的工具调用、工具结果、答案分片、引用和结束状态拆成事件
  > 推给前端，从而支持大模型回答的流式展示。

```
项目中体现：

- `app/main.py` 创建 FastAPI 应用。
- `app/api/routes/chat.py` 提供 `/api/v1/chat` 和 `/api/v1/chat/stream`。
- `app/api/routes/document.py` 提供文档上传和列表接口。

#### LangGraph

为什么用 LangGraph：

- 普通 LLM 调用是“输入 -> 输出”，不适合表达复杂 Agent 流程。
- LangGraph 可以把 Agent 拆成状态图：路由、工具执行、回答生成、澄清问题。
- 每个节点职责明确，更容易调试、扩展和面试讲解。

项目中主链路：

```text
START
  -> pre_tool_router
  -> tool_executor 或 answer_generator 或 clarify
  -> answer_generator
  -> END

START：入口节点，收到用户问题后进入流程。

pre_tool_router：前置路由器，判断问题类型和是否需要澄清，决定要不要调用工具以及调用哪些工具。

tool_executor 或 answer_generator 或 clarify：分支节点

tool_executor：当需要工具时执行（例如 RAG 检索、Verilog 分析、时序约束建议）。

answer_generator：当不需要工具或工具为空时，直接生成回答（可能是严格拒答）。

clarify：问题过短/不清晰时，返回澄清提示，让用户补充信息。

answer_generator：汇总工具结果，生成最终回答（或严格拒答）。

END：流程结束，返回给用户。

 IC 垂类 LangGraph 主链路：
    1) pre_tool_router：根据意图决定工具；
    2) tool_executor：执行工具并收集输出；
    3) answer_generator：将工具结果注入上下文，生成最终回复。
    """
```

#### LlamaIndex + Chroma

为什么用 LlamaIndex：

- 它提供文档、节点、索引、retriever 的高层抽象，适合快速搭建 RAG。
- 可以和向量数据库、embedding 模型集成。

为什么用 Chroma：

- 轻量、可本地持久化，适合 demo、课程项目和个人项目展示。
- 不需要部署复杂向量数据库，降低运行成本。

项目中体现：

- `app/core/rag/retriever.py` 中 `ICRAGRetriever` 使用 `ChromaVectorStore` 和 `VectorStoreIndex`。
- 向量库默认落在 `chroma_db`。

#### HuggingFace Embedding + CrossEncoder Reranker

为什么用 embedding：

- 把 IC 文档切块后编码成向量，支持语义检索，而不是只能关键词匹配。

除了语义检索（dense/vector），常见的检索方式还有：

关键词/词项检索（lexical），典型是 TF-IDF、BM25

布尔检索（AND/OR/NOT），更偏精确匹配

结构化检索（按字段/元数据过滤，比如 source、page、章节）

规则/模式检索（正则、模板匹配）

图检索/知识图谱检索（基于实体关系）

混合检索（Hybrid：BM25 + 向量）
### 语义检索的优点：
能处理同义表达和语序变化，避免“关键词不一致就搜不到”
更适合自然语言提问（比如“乘法器优化”能召回“Booth/Wallace”相关内容）
对短问题更鲁棒，减少纯关键词的漏召回
在这个项目里，IC 文档很多概念表达不统一，语义检索能提高召回率；再用 reranker 提精度。

### 为什么加 reranker：

- 第一阶段向量检索召回可能包含噪声。
- CrossEncoder 可以对 query-document pair 重新打分，让 top-k 结果更精确。

项目中体现：

- `ICRAGRetriever` 默认 embedding 模型是 `BAAI/bge-m3`。
- `app/core/rag/reranker.py` 使用 `sentence_transformers.CrossEncoder`。

#### Pydantic / SQLAlchemy / Redis 等基础设施

这些依赖体现了项目有“企业级 Agent 服务”的雏形：

- Pydantic：定义请求、响应、RAG 结果、引用等 schema。
- SQLAlchemy：建模 Conversation、Message、Document、DocumentChunk、TraceLog。
- Redis：保留缓存能力。
- Tracer：记录 Agent 调用链路。

面试时不要把这些说成“都深度用了”，而要准确表达：

> 当前主链路重点在 FastAPI + LangGraph + RAG + 工具调用，数据库、缓存、trace 是为了让项目具备服务化和可观测扩展能力。

---

### 3. 系统整体架构

可以按五层来讲。

```text
用户 / 前端 / curl
        |
        v
API 接入层 FastAPI
- /api/v1/chat
- /api/v1/chat/stream
- /api/v1/documents/upload
        |
        v
Agent 编排层 LangGraphICAgent
- pre_tool_router
- tool_executor
- answer_generator
- clarify
        |
        v
领域工具层 IC Tools
- ic_rag_search
- verilog_code_analyzer
- timing_constraint_suggester
        |
        v
RAG / 数据层
- PDF / TXT 文档
- IC 定制分块
- HuggingFace Embedding
- Chroma 向量库
- LlamaIndex Retriever
- CrossEncoder Reranker
        |
        v
模型与基础设施层
- ModelRouter
- OpenAI-compatible LLM
- Tracer
- SQLAlchemy models
- Evaluation scripts
```

#### 前端 / 客户端层

当前项目已经有内置 Web 前端，主要通过：

- 浏览器访问 `/`
- HTTP 非流式接口 `/api/v1/chat`
- SSE 流式接口 `/api/v1/chat/stream`
- 强自主任务接口 `/api/v1/agent/run`
- curl / API client 调用

Web 前端
由 `app/static/index.html`、`app/static/app.js`、`app/static/styles.css` 组成，不需要 React/Vue 构建。页面包含：

- 普通聊天区。
- 会话 ID 和记忆状态。
- 工具调用面板。
- 来源引用面板。
- 自主任务面板。
- 任务步骤、置信度、证据数、复核标记展示。

HTTP 非流式接口 `/api/v1/chat`
普通的 POST 接口，服务端一次性返回完整回答。适合简单调用或后端对接。响应里会返回 `conversation_id`，用于后续复用记忆。

SSE 流式接口 `/api/v1/chat/stream`
用 Server-Sent Events 持续推送响应，客户端可以边接收边显示，固定事件包括 `tool_call`、`tool_result`、`answer`、`citation`、`done`、`error`。

强自主任务接口 `/api/v1/agent/run`
一次性执行一个目标任务，返回完整任务轨迹。适合展示“Agent 不只是问答，而是能拆计划、调用工具、恢复失败、反思审查”。

curl / API client
仍然可以用命令行、Postman、Apifox、Python/JS 调用上述接口。

#### 后端 API 层

核心职责：

- 接收用户消息。
- 读取短期和长期记忆。
- 构造 `ModelRouter`。
- 构造 IC 工具注册表。
- 构造 `LangGraphICAgent`。
- 执行 Agent。
- 调用引用重写。
- 保存用户与助手消息到记忆。
- 返回结构化响应。
- 对强自主任务，构造 `AutonomousAgent` 并返回完整任务轨迹。

关键文件：

- `app/main.py`
- `app/api/routes/chat.py`
- `app/api/routes/agent.py`
- `app/api/routes/document.py`
- `app/api/routes/health.py`
```
ModelRouter 是这个项目里的“大模型调用层路由器”，不是 Agent 的工具路由器。

  它的位置在 app/infrastructure/llm/model_router.py:41。

  它主要做几件事：

  1. 统一封装模型调用
     把 OpenAI 兼容接口封装成 chat(messages, ...)，返回统一的 LLMResponse，里面有 content、model_id、usage、raw。
  2. 选择模型
     支持根据 priority 优先级、weight 权重选择模型。如果传了 model_preference，会优先找指定模型。
  3. 失败降级
     如果某个模型调用失败，会尝试下一个候选模型。相关逻辑在 app/infrastructure/llm/model_router.py:95。
  4. 熔断保护
     每个模型都有一个 CircuitBreaker。如果连续失败达到阈值，就临时跳过这个模型，避免一直打坏掉的服务。熔断器在 app/infrastructure/llm/circuit_breaker.py:24。

  在请求链路里，它是这样用的：

  chat.py 里先用配置构造 ModelConfig，再创建 ModelRouter：

  app/api/routes/chat.py:27

  然后把它传给 LangGraphICAgent：

  app/api/routes/chat.py:42

  最后在 Agent 的 answer_generator 阶段调用：

  app/core/agent/langgraph_agent.py:358

  可以这样面试表达：

  > ModelRouter 是我在 LLM 调用层做的一层抽象，负责统一模型调用、模型选择、失败降级和熔断保护。Agent 不直接依赖具体模型 API，而是通过 ModelRouter.chat() 调用模
  > 型，这样后续切换模型、增加备用模型、配置权重和优先级都不需要改 Agent 主流程。

  但注意一点：当前项目实际只配置了一个模型，因为 _build_router() 里只创建了一个 ModelConfig。所以多模型路由、权重负载、降级能力是架构上支持了，但还没有真正配置多
  个模型池。
```

#### Agent 编排层

核心职责：

普通问答 Agent：

- 判断用户问题是否需要澄清。
- 判断是否属于 IC 领域。
- 判断要调用哪些工具。
- 把工具结果组织成上下文交给大模型。
- 在证据不足时严格拒答。

强自主 Agent：

- 根据目标生成计划。
- 按步骤执行工具或推理。
- 捕获工具失败并降级恢复。
- 收集 evidence、confidence、review_flags。
- 生成最终交付。
- 调用反思 Agent 做质量审查。
- 生成任务级 audit_summary。

关键文件：

- `app/core/agent/langgraph_agent.py`
- `app/core/agent/autonomous.py`
- `app/core/agent/reflection.py`

#### 工具层

核心职责：

- 把不同能力封装成可调用工具。
- Agent 不直接关心工具内部实现，只关心工具名和参数。
- 工具调用前做参数 schema 校验。
- 工具结果统一归一化为 `ok/result/summary/evidence/confidence/review_flags`。

核心工具：

- `ic_rag_search`：IC 知识检索。
- `verilog_code_analyzer`：Verilog 规则型检查。
- `timing_constraint_suggester`：SDC 时序约束建议。

关键文件：

- `app/core/tools/builtin/ic_tools.py`
- `app/core/tools/registry.py`
- `app/core/tools/factory.py`
- `app/core/tools/base.py`

#### 数据层 / 检索层

核心职责：

- 从 `data/` 目录读取 PDF。
- 用 IC 定制分块器切块。
- 写入或读取 Chroma。
- 检索 top-k 文档。
- 返回带 source/page/chunk_id 的结构化结果。
- 管理 JSONL 短期会话历史。
- 管理 JSONL 长期记忆，并用简单关键词重叠召回。
- 可选使用 Milvus 替换长期记忆召回。

关键文件：

- `app/core/rag/retriever.py`
- `app/core/rag/reranker.py`
- `app/core/rag/citation_rewriter.py`
- `app/core/memory/short_term.py`
- `app/core/memory/long_term.py`
- `app/core/memory/local.py`
- `app/core/memory/milvus.py`
- `app/core/memory/manager.py`
- `app/core/memory/factory.py`
- `app/etl/ic_text_splitter.py`

---

### 4. 数据流是如何走的？

以一次 `/api/v1/chat` 请求为例。

```text
1. 用户发送问题
   |
2. FastAPI chat endpoint 接收 ChatRequest
   |
3. 根据 conversation_id 读取短期历史和长期记忆
   |
4. 构造 LangGraphICAgent
   |
5. pre_tool_router 分析问题
   |-- 太短：进入 clarify
   |-- 非 IC：不调用工具，后续严格拒答
   |-- IC 知识：选择 ic_rag_search
   |-- Verilog 代码：选择 ic_rag_search + verilog_code_analyzer
   |-- 时序约束：选择 ic_rag_search + timing_constraint_suggester
   |
6. tool_executor 执行工具
   |
7. 工具注册中心校验参数，并把工具结果归一化为审计记录
   |
8. ic_rag_search 调用 ICRAGRetriever
   |
9. Retriever 从 Chroma / LlamaIndex 检索文档，必要时 rerank
   |
10. 工具结果被塞进 answer_generator 的上下文
   |
11. LLM 严格基于工具结果生成回答
   |
12. citation_rewriter 删除模型自造引用，重建服务端引用
   |
13. 保存本轮 user/assistant 到记忆
   |
14. 返回 ChatResponse：answer、conversation_id、trace_id、sources、tool_events
```

面试中可以强调：

> 这个系统的数据流不是“问题直接给大模型”，而是先经过领域路由，再经过工具执行，再由模型基于工具证据回答，最后服务端治理引用。

以一次 `/api/v1/agent/run` 请求为例。

```text
1. 用户提交目标 goal
   |
2. FastAPI agent endpoint 创建或复用 conversation_id
   |
3. AutonomousAgent 先把目标写入记忆
   |
4. _plan 使用 LLM 规划；LLM 不可用时走启发式计划
   |
5. 逐步执行 steps
   |-- tool step：构造工具参数，调用 invoke_with_audit
   |-- reasoning step：调用 LLM 推理，失败时降级为保守观察
   |
6. 每一步记录 rationale、arguments、observation、evidence、confidence、review_flags
   |
7. 工具失败时记录 error，并用 _recover 降级恢复
   |
8. _finalize 汇总执行轨迹、证据边界和下一步
   |
9. _reflect 做反思审查；反思模型不可用时保守评分
   |
10. _build_audit_summary 生成任务级审计摘要
   |
11. 根据 error/review_flags/confidence/reflection 推导状态
   |-- completed
   |-- needs_review
   |-- failed
   |
12. 保存最终结果到记忆，返回 AutonomousTask
```

面试中可以强调：

> `/agent/run` 和普通问答最大的区别是它不是单轮回答，而是“目标 -> 计划 -> 执行 -> 恢复 -> 反思 -> 审计”的任务闭环。即使工具失败，它也会保留错误和复核标记，不会把低证据结果包装成确定结论。

---

## 第二阶段：核心模块深度拆解（技术掌控）

### 模块一：API 路由层

关键文件：`app/api/routes/chat.py`

#### 存在的必要性

API 层负责把外部请求转换成内部 Agent 调用。它不是简单转发，而是承担了：

- 请求 schema 校验。
- trace_id 生成。
- Agent 构造。
- 非流式和流式两种返回形态。
- 引用重写。
- 错误处理。

请求 schema 校验：检查用户发来的请求格式是否正确，比如有没有messages、字段类型对不对；不对就直接报错，避免后面出错。

trace_id 生成：给每次请求一个唯一编号，方便日志追踪和排查问题（比如一条请求从进来到回答的全流程）。

Agent 构造：为这次请求创建并准备好 Agent（包括模型路由器、工具注册表、LangGraph 主链路），让它能执行你的问题。

非流式强调“简单稳定”，流式强调“实时体验”，两者兼顾不同用户和场景。
#### 核心逻辑

伪代码：

```python
async def chat(request):
    trace_id = uuid4()
    span = tracer.start_trace(trace_id, "chat")

    messages = request.messages
    agent = build_ic_agent()

    result = await agent.run(messages)
    rewritten = rewrite_answer_citations(result.content, result.sources)

    tracer.end_span(span, result={...})

    return ChatResponse(
        answer=rewritten.answer,
        sources=result.sources,
        tool_events=result.tool_events,
        trace_id=trace_id,
    )
```

SSE 流式接口的逻辑：

```text
先完整跑完 Agent
再按事件顺序吐出：
- tool_call
- tool_result
- answer chunk
- citation
- done
```

#### 为什么当前这样做？

优点：

- 实现简单，非流式和流式复用同一套 Agent 结果。
- SSE 事件结构清晰，方便前端展示工具调用过程。
- trace 和 citation 都在路由层统一处理。

不足：

- 当前 SSE 不是真正 token-level streaming，而是 Agent 完整执行后再分块输出 answer。
- 每次请求都会构造 agent 和 tool registry，可能有重复初始化成本。

#### 如果重写，会怎么设计？

可以改成：

- 把 `ModelRouter`、`ToolRegistry`、`LangGraphICAgent` 做成应用生命周期内的 singleton，避免每次请求重复构造。
```
现在代码里每次用户请求 /chat，都会重新创建一套对象：

  ModelRouter：模型调用器
  ToolRegistry：工具注册表
  LangGraphICAgent：Agent 主流程图

  也就是每次请求都走一遍：

  agent = _build_ic_agent()

  这里会重新构造 ModelRouter、重新构造工具、重新构造 LangGraphICAgent。位置在 app/api/routes/chat.py:42。

  “做成应用生命周期内的 singleton” 的意思是：

  > FastAPI 服务启动时只创建一次这些对象，之后所有请求复用同一套对象，而不是每次请求重新 new。

  比如：

  # 服务启动时
  app.state.model_router = ModelRouter(...)
  app.state.tool_registry = build_ic_tool_registry()
  app.state.ic_agent = LangGraphICAgent(
      model_router=app.state.model_router,
      tool_registry=app.state.tool_registry,
  )

  # 每次请求时
  agent = request.app.state.ic_agent

  这样做的好处是：

  1. 减少重复初始化开销
     比如 AsyncOpenAI client、工具注册表、LangGraph graph 编译、RAG retriever 初始化等，不用每次请求都重新创建。
  2. 提升性能
     请求来了直接跑 agent.run()，不用先构造一堆基础设施对象。
  3. 更符合服务端工程习惯
     这些对象本质上是基础设施，不是用户请求数据，应该随着应用启动而创建，随着应用关闭而释放。

  但要注意：

  singleton 不等于共享用户对话状态。

  正确做法是：

  - ModelRouter 可以共享；
  - ToolRegistry 可以共享；
  - LangGraphICAgent 可以共享；
  - 每次请求的 messages、trace_id、tool_outputs、AgentState 仍然是独立的。

  面试可以这样说：

  > 目前项目中 ModelRouter、ToolRegistry 和 LangGraphICAgent 是在每次请求内构造的，功能上没问题，但会带来重复初始化开销。我会进一步把这些无状态基础设施对象挂到
  > FastAPI 的 app.state 或 lifespan 中，在应用启动时初始化一次，请求时复用，从而降低延迟并提升吞吐。

```
- 如果模型 SDK 支持 streaming，把 `answer_generator` 改成真正边生成边推送。
- 把 trace span 细化到每个 Agent 节点：router、tool、retriever、llm。

面试追问：为什么不一开始就这么做？

回答：

> 因为这是一个实习展示项目，第一优先级是跑通主链路和保证结果可靠。当前实现先保证结构清晰，后续性能优化可以在压测后再做，比如复用 agent、连接池、真正 token streaming。

---

### 模块二：LangGraphICAgent 编排层

关键文件：`app/core/agent/langgraph_agent.py`

#### 存在的必要性

如果没有 Agent 编排，用户问题只能直接丢给大模型，系统无法控制：

- 哪些问题应该检索。
- 哪些问题应该调用 Verilog 分析工具。
- 哪些问题应该生成时序约束。
- 哪些问题应该澄清。
- 哪些问题应该拒答。

LangGraphICAgent 的价值是把“怎么回答”拆成显式流程，而不是依赖模型自由发挥。

#### 核心状态

`AgentState` 里包含：

- `user_query`：用户最后一条问题。
- `messages`：完整消息。
- `selected_tools`：路由选择的工具。
- `tool_outputs`：工具执行结果。
- `final_answer`：最终回答。
- `sources`：从工具结果中抽取出的引用来源。
- `route_reason`：路由原因。
- `needs_clarification`：是否需要澄清。

#### 核心流程

```text
pre_tool_router
  1. 空问题 -> 澄清
  2. 过短问题，比如“乘法器”“时序” -> 澄清
  3. 非 IC 领域 -> 不调用工具，后续严格拒答
  4. IC 知识问题 -> ic_rag_search
  5. Verilog 代码问题 -> ic_rag_search + verilog_code_analyzer
  6. 时序约束问题 -> ic_rag_search + timing_constraint_suggester

route_after_pre_tool_router
  - needs_clarification=True -> clarify
  - selected_tools 非空 -> tool_executor
  - 否则 -> answer_generator

tool_executor
  - 根据工具名构造参数
  - 调用 tool_registry.invoke
  - 收集结果

answer_generator
  - 无工具结果/检索未命中 -> 严格拒答
  - 有工具结果 -> 拼上下文调用 LLM
```

#### 有没有更优解？

当前路由是规则型路由，优点是：

- 可控。
- 可解释。
- 适合面试展示。
- 不会因为 LLM 路由不稳定导致乱调工具。
``` 
 规则型路由就是：不用大模型判断，而是用代码里的固定规则判断用户问题该走哪条流程、调用哪些工具。

  在你这个项目里，规则型路由主要就是 pre_tool_router，位置在 app/core/agent/langgraph_agent.py:192。

  它会根据关键词、问题长度、领域词来判断：

  - 问题太短，比如“乘法器”“时序”
    返回澄清问题，不直接回答。
  - 不是 IC/Verilog 领域
    不调用工具，后面严格拒答。
  - 是 IC 知识类问题
    调用 ic_rag_search。
  - 包含 Verilog/RTL/code 等关键词，并且有代码
    调用 verilog_code_analyzer。
  - 包含 SDC、setup、hold、clock、时序等关键词
    调用 timing_constraint_suggester。

  可以理解成：

  if 问题太短:
      让用户补充
  elif 不是 IC 领域:
      拒答
  elif 是知识问答:
      调 RAG
  elif 是 Verilog 代码:
      调代码分析工具
  elif 是时序约束:
      调 SDC 工具

```
缺点是：

- 关键词规则覆盖有限。
- 领域扩展后维护成本会上升。
- 对复杂混合意图的理解不如模型路由。

更优方案：

- 短期：继续用规则路由，但把关键词、工具描述、阈值配置化。
- 中期：加入轻量意图分类器，输出 `knowledge/verilog/timing/out_of_scope/clarify`。
- 长期：规则路由兜底 + LLM function calling 路由，两者投票或优先级结合。

#### 如果重写，会怎么设计？

可以把路由层拆成独立模块：

```text
IntentRecognizer
  -> returns IntentResult
       - domain: ic / non_ic
       - task_type: rag / code_review / timing / mixed
       - confidence
       - selected_tools
       - reason
```

然后 Agent 只负责状态流转，不直接维护大量关键词。

但当前项目用于实习展示时，可以这样讲：

> 我现在用规则路由是为了保证可解释和稳定，特别是 IC 领域 demo 中，错误调用工具比少调用工具更影响体验。后续如果数据量和意图类型变多，我会把规则抽成配置，并引入意图分类模型做增强。

---

### 模块三：IC 工具层

关键文件：`app/core/tools/builtin/ic_tools.py`

#### 存在的必要性

Agent 的核心思想是“模型不是什么都自己做”，而是把确定性、可复用、领域化能力封装成工具。

这个项目中工具层解决三个问题：

1. 知识问答要查文档，而不是靠模型记忆。
2. Verilog 代码检查适合规则分析，不一定要完全交给 LLM。
3. 时序约束建议可以模板化生成，降低用户从 0 写 SDC 的成本。

---

#### 工具一：ICRAGSearchTool

##### 必要性

IC 专业知识非常细，例如 Verilog 语法、时序分析、乘法器结构、综合限制。直接问大模型容易出现：

- 回答不基于项目资料。
- 引用不存在。
- 通用知识和本地文档不一致。

所以需要 RAG 检索工具。

##### 核心逻辑

```python
query = user_query
expanded_query = expand_ic_query(query)
results = retriever.retrieve(expanded_query, top_k * 3)
ranked = rank_results_by_terms(results, query)
return top_k snippets with source/page/chunk_id
```

它不是裸向量检索，还做了：

- query expansion：比如“乘法器”扩展成 Booth、Wallace、阵列乘法器、关键路径、时序优化等。
- focus keyword：根据 query 加强领域关键词。
- snippet extraction：从长 chunk 中截取和问题最相关的句子。
- result dedup/rank：按 source/page/chunk_id 去重并重新排序。

##### 为什么当前这样做？

因为用户问题可能很短，比如“乘法器优化”，直接 embedding 检索可能召回不稳。轻量 query expansion 可以显著提升召回质量。

##### 更优解

- 用 HyDE 生成假设答案再检索。
```
 HyDE 全称是 Hypothetical Document Embeddings，中文可以理解成：假设性文档嵌入。

  它是 RAG 检索优化方法。

  普通 RAG 是：

  用户问题 -> embedding -> 去向量库里搜相似 chunk

  HyDE 是多走一步：

  用户问题 -> 让大模型先生成一段“假想答案/假想文档”
  假想文档 -> embedding -> 去向量库里搜相似 chunk

  为什么这么做？

  因为用户问题通常很短，信息量少。比如：

  乘法器怎么优化？

  这个 query 太短，直接 embedding 可能召回不准。

  HyDE 会先让模型生成一段更像知识库文本的内容，例如：

  乘法器时序优化通常包括流水线设计、减少组合逻辑路径、使用 Booth 编码、 Wallace Tree 压缩、寄存器平衡等方法……

  然后用这段“假想文档”去做向量检索。它更接近文档表达方式，召回效果可能更好。

  核心思想：

  > 与其拿短 query 去匹配长文档，不如先把 query 扩写成一段像文档的文本，再拿它去检索。

  优点：

  - 提高短问题、模糊问题的召回率；
  - 让 query 更接近知识库文档风格；
  - 对专业领域 RAG 有帮助。

  缺点：

  - 多一次 LLM 调用，成本和延迟更高；
  - 如果假想文档生成偏了，可能把检索方向带歪；
  - 不能直接相信 HyDE 生成的内容，它只是用来检索，不是最终答案。

  在面试里可以这样说：

  > HyDE 是一种 RAG 查询增强方法。它不是直接把用户问题向量化，而是先让大模型基于问题生成一段假想答案或假想文档，再对这段文本做 embedding 去向量库检索。这样可以缓
  > 解用户 query 太短、和知识库表达不一致的问题，提高召回率。但 HyDE 生成内容本身不能作为事实依据，只能作为检索 query，最终回答仍然要基于真实召回的文档片段。

 
```
- 加 BM25 + dense retrieval 的 hybrid search。
```
• BM25 + dense retrieval 是一种 混合检索 Hybrid Search。

  它把两种检索方式结合起来：

  BM25 关键词检索 + Dense Retrieval 向量语义检索

  ## 1. BM25 是什么？

  BM25 是传统搜索引擎常用的关键词检索算法。

  它看的是：

  - query 里的词有没有在文档里出现；
  - 出现频率高不高；
  - 这个词是不是稀有词；
  - 文档长度是否合适。

  比如用户问：

  setup hold violation

  BM25 很擅长找出明确包含 setup、hold、violation 这些词的文档。

  优点：

  - 对关键词、术语、代码符号、型号、函数名很准；
  - 不需要训练模型；
  - 可解释性强。

  缺点：

  - 不懂语义；
  - 用户换一种说法可能搜不到。

  ## 2. Dense Retrieval 是什么？

  Dense Retrieval 是向量检索。

  它会把 query 和文档 chunk 都转成 embedding 向量，然后按向量相似度检索。

  比如：

  时钟太快导致寄存器采样失败怎么办？

  即使文档没有完全一样的词，dense retrieval 也可能搜到：

  setup time violation

  优点：

  - 能理解语义相似；
  - 适合自然语言问答；
  - 对同义表达更友好。

  缺点：

  - 对精确关键词、代码符号、缩写有时不如 BM25；
  - 可能召回“语义像但关键词不对”的内容。

  ## 3. 为什么要合起来？

  因为 IC/Verilog 项目里既有自然语言，也有大量精确术语：

  - always
  - assign
  - posedge
  - setup
  - hold
  - SDC
  - fork/join
  - nonblocking assignment
  - 模块名、信号名、参数名

  只用 dense retrieval，可能语义对但漏掉关键符号。
  只用 BM25，可能关键词对但不懂中文语义问题。

  所以混合检索是：

  用户 query
    -> BM25 找关键词匹配 chunk
    -> Dense Retrieval 找语义相似 chunk
    -> 合并候选结果
    -> 去重
    -> reranker 重排
    -> 返回 top-k

  ## 4. 面试怎么说？

  你可以这样说：

  > BM25 + dense retrieval 是混合检索。BM25 负责关键词和术语级召回，适合 Verilog 关键字、SDC 命令、信号名这类精确匹配；dense retrieval 负责语义召回，适合用户用自
  > 然语言描述问题。两路召回后再合并去重，并交给 reranker 重排，能同时提升召回率和准确性。

  ## 5. 和你项目的关系

  你当前项目主要是：

  dense retrieval + reranker

  也就是：

  Chroma 向量召回 -> CrossEncoder reranker 重排

  还没有完整做 BM25 + dense retrieval 混合检索。

  如果要升级，可以加：

  BM25 召回 topN
  Chroma dense 召回 topN
  合并去重
  CrossEncoder rerank
  返回 topK
```
- 对 query 做结构化解析，比如 topic=multiplier，aspect=timing optimization。

---

#### 工具二：VerilogCodeAnalyzerTool

##### 必要性

Verilog 代码中很多问题是规则型的，比如：

- 组合逻辑 latch 风险。
- 阻塞/非阻塞赋值混用。
- always 敏感列表问题。
- 复位风格问题。

这些问题不一定需要大模型推理，规则检查更稳定、可解释。

##### 核心逻辑

```text
输入 Verilog 代码
  -> 正则/规则扫描
  -> 检查 always、assign、reset、blocking/non-blocking
  -> 输出 warnings / suggestions
```

##### 为什么当前这样做？

- 规则型工具成本低。
- 对 demo 来说稳定可控。
- 能体现 Agent “调用专业工具”的能力。

##### 更优解

- 接入 Verilator、iverilog 或 yosys 做真正语法检查。
- 建 AST，而不是只用正则。
- 把 lint rule 分级：error/warning/style。

面试回答：

> 当前版本偏轻量规则引擎，用于展示 Agent 工具调用能力。如果要工程化，我会接 Verilator 或 Surelog 这类工具，把语法解析、lint、综合可行性检查做得更准确。

---

#### 工具三：TimingConstraintSuggesterTool

##### 必要性

SDC 约束对初学者很难，尤其是：

- create_clock 怎么写。
- input_delay / output_delay 怎么估。
- false path / multicycle path 怎么表达。
- setup/hold 相关约束怎么理解。

这个工具把常见时序约束模式模板化，帮助用户快速得到草案。

##### 核心逻辑

```text
从问题中猜 clock period
根据 module / clock / reset 信息
生成 SDC 约束模板
附带解释和注意事项
```

##### 当前方案优缺点

优点：

- 快速、确定、适合教学。
- 不依赖复杂 EDA 环境。

缺点：

- 无法知道真实网表、时钟树、IO 时序环境。
- 生成的是建议，不是可直接 signoff 的约束。

面试中要主动说明边界：

> 这个工具生成的是 SDC 初稿，不是最终签核约束。真实项目里需要结合芯片架构、时钟域、IO spec 和 STA 报告进一步修改。

---

### 模块四：RAG 检索层

关键文件：`app/core/rag/retriever.py`

#### 存在的必要性

RAG 检索层是项目可信回答的基础。如果检索层不可靠，后面的回答生成和引用治理都会失去意义。

它解决的问题：

- 如何从本地 IC 文档构建知识库。
- 如何保证向量库和 data 目录一致。
- 如何返回带 source/page 的可追溯结果。
- 如何在初检后重排，提高相关性。

#### 核心逻辑

```python
class ICRAGRetriever:
    def retrieve(query, top_k):
        index = ensure_index()
        raw_results = index.as_retriever(similarity_top_k=candidate_k).retrieve(query)
        if reranker enabled:
            results = reranker.rerank(query, raw_results)
        return [ICRetrievalResult(content, source, page, score, chunk_id)]
```

#### 索引维护逻辑

```text
ensure_index
  -> 连接 Chroma collection
  -> 检查 data 目录 PDF 和 collection source 元数据是否一致
  -> 如果一致：加载已有 index
  -> 如果不一致且 strategy=rebuild：重建 collection
  -> 如果不一致且 strategy=warn：只告警
```

这个设计很值得讲。

面试表达：

> 我在检索器里做了 source consistency check，会比较 data 目录里的 PDF 文件集合和 Chroma collection 里的 source 元数据。如果用户换了知识库文件，但向量库没有更新，系统可以发现不一致，并根据配置选择 warn 或 rebuild，避免检索到旧文档导致引用错误。

#### 结构化返回

每条检索结果包含：

- `content`：文本内容。
- `source`：来源文件。
- `page`：页码。
- `score`：检索或重排分数。
- `chunk_id`：块 ID。

为什么重要：

- answer_generator 可以基于 content 回答。
- citation_rewriter 可以基于 source/page 重写引用。
- evaluation 可以评估 citation correctness。

#### 更优解

当前是 dense retrieval + optional reranker。更强方案：

- hybrid retrieval：BM25 + dense。
- metadata filtering：按文档类型、章节、页码过滤。
- parent-child chunking：小块检索，大块喂模型。
- incremental indexing：上传单个文档后只增量更新，不重建全部索引。

---

### 模块五：IC 定制分块 ETL

关键文件：`app/etl/ic_text_splitter.py`

#### 存在的必要性

RAG 项目里，chunking 质量直接影响检索质量。

普通固定长度切块会破坏 IC 文档里的结构，例如：

- 把 `module ... endmodule` 切断。
- 把章节标题和正文分开。
- 把时序图说明和解释分开。
- 把 Verilog 代码块拆碎。

所以项目实现了 `ICCustomTextSplitter`。

#### 核心逻辑

```text
输入文档文本
  -> 优先按 Verilog module/endmodule 切
  -> 再按章节标题、时序图、波形图切
  -> 如果块太长，再交给 RecursiveCharacterTextSplitter 细切
  -> 保留 metadata，比如 source/page
```

#### 为什么当前这样做？

因为 IC 文档有明显领域结构：

- Verilog 代码需要完整上下文。
- 时序图和解释需要放在一起。
- 章节边界通常代表语义边界。

这比纯 token chunk 更适合 IC 知识问答。

#### 更优解

- 对 PDF 版面做 layout-aware parsing。
- 识别表格、图片 caption、代码块。
- 对 Verilog 代码用 parser 抽取 AST 或 module-level chunk。
- 建立 chunk hierarchy：章节 -> 小节 -> 代码块。

面试表达：

> 我没有直接用固定窗口切块，而是根据 IC 文档特点保留 Verilog module、章节和时序图边界。这样检索到的 chunk 更完整，后续回答更不容易断章取义。

---

### 模块六：引用重写 Citation Rewriter

关键文件：`app/core/rag/citation_rewriter.py`

#### 存在的必要性

RAG 系统最大的风险之一是“引用幻觉”：

- 模型可能编造参考资料。
- 模型可能把未检索到的 source/page 写进回答。
- 用户看到引用后会误以为可信。

所以项目把“最终引用权”从模型手里拿回来，交给服务端。

#### 核心逻辑

```python
rewrite_answer_citations(answer, sources):
    refs = build_reference_entries(sources)
    body = strip_model_reference_section(answer)
    body = remove_fake_inline_citations(body, refs)
    ref_block = render_reference_block(refs)
    return body + ref_block
```

具体做了三件事：

1. 删除模型自己生成的“参考资料/References”区块。
2. 检查正文中 `来源: xxx | 页码: yyy` 这类引用是否存在于本轮 sources。
3. 由服务端重新生成：

```text
参考资料（服务端生成）
1. xxx.pdf | 第12页
2. yyy.pdf | 第8页
```

#### 为什么当前这样做？

因为 prompt 约束不能 100% 阻止模型编引用。服务端后处理是更强约束。

#### 更优解

- 在回答中做句子级 citation alignment。
- 每个段落绑定 evidence chunk。
- 如果某段没有 evidence，直接删除或标记“无引用支撑”。
- 在 UI 中点击引用可跳转 PDF 页码。

面试表达：

> 我把 citation 当成服务端安全边界处理，而不是完全相信模型输出。模型只负责回答，最终引用列表由服务端根据真实检索结果生成，这样可以显著降低伪引用风险。

---

### 模块七：模型路由 ModelRouter

关键文件：`app/infrastructure/llm/model_router.py`

#### 存在的必要性

AI 服务不能把模型调用散落在业务代码里，需要统一封装：

- 模型选择。
- API key / base url。
- fallback。
- 熔断。
```

• 熔断就是：某个下游服务连续失败太多次，系统先暂停调用它一段时间，避免每个请求都继续撞上失败。

  在你这个项目里，下游服务主要指 大模型接口。

  比如模型服务连续报错：

  超时
  限流
  API error
  网络失败

  如果没有熔断，每个用户请求都会继续调用这个坏掉的模型，结果是：

  请求变慢 -> 一直报错 -> 服务堆积 -> 整个系统被拖垮

  有熔断后：

  模型连续失败达到阈值
  -> 熔断器打开 OPEN
  -> 后续请求先跳过这个模型
  -> 等一段时间
  -> 进入 HALF_OPEN 试探
  -> 如果试探成功，恢复 CLOSED
  -> 如果还失败，继续 OPEN

  三个状态：

  CLOSED：正常调用，记录失败次数
  OPEN：熔断打开，直接拒绝调用
  HALF_OPEN：过一段时间后，放少量请求试探服务是否恢复

  你项目里的实现位置是 app/infrastructure/llm/circuit_breaker.py:24。

  它被 ModelRouter 使用。ModelRouter 给每个模型都配一个熔断器：

  app/infrastructure/llm/model_router.py:55

  面试可以这样说：

  > 熔断是一种保护下游服务和自身系统稳定性的机制。当某个模型接口连续失败达到阈值后，熔断器会进入 OPEN 状态，短时间内不再调用这个模型，而是快速失败或切换到其他候
  > 选模型。等待恢复时间后进入 HALF_OPEN 状态，用少量请求试探服务是否恢复，成功则关闭熔断，失败则继续打开。这个项目里 ModelRouter 给每个模型配置了
  > CircuitBreaker，用于模型调用失败时的保护和降级。
```
- usage 统计。

#### 当前项目中的角色

`chat.py` 中 `_build_router()` 当前只配置了一个模型，但 `ModelRouter` 的设计支持多模型候选。

这说明：

- 当前 demo 简单。
- 底层架构保留了扩展空间。

面试要准确表达：

> 当前主链路默认只用一个 OpenAI-compatible 模型配置，但 ModelRouter 这个抽象可以扩展到多模型优先级、加权选择、失败降级和熔断。

#### 更优解

- 根据任务选择模型：路由用小模型，最终回答用大模型。
- RAG 问答用低温度，代码解释用中低温度。
- 对失败类型分类：限流、超时、内容过滤、网络错误。

---

### 模块八：评测闭环

关键文件：`evaluation/evaluate_ragas.py`

#### 存在的必要性

很多 RAG 项目只做到“看起来能回答”，但没有验证质量。这个项目加入评测后，可以回答面试官的问题：

> 你怎么知道你的 Agent 是有效的？

#### 核心逻辑

```text
读取 evaluation/datasets/eval_dataset_qg_30.json
  -> 对每个问题调用最终 chat 主链路
  -> 得到 answer / sources / tool_events
  -> citation_rewriter 后处理
  -> 计算 RAGAS 指标
  -> 额外计算 citation_correctness / refusal_correctness / tool_routing_accuracy
  -> 输出 reports
```

#### 评测指标

RAGAS 指标：

- `faithfulness`：回答是否忠实于上下文。
- `answer_relevancy`：回答是否切题。
- `context_recall`：检索上下文是否覆盖答案所需信息。
- `context_precision`：检索上下文是否足够精确。

项目自定义指标：

- `citation_correctness`：引用是否来自真实检索结果。
- `refusal_correctness`：该拒答时是否拒答。
- `tool_routing_accuracy`：工具路由是否符合预期。
```
groundtruth 就是标准答案 / 标注答案 / 参考真值。

  在 RAG 评估里，它表示：这个问题理想情况下应该回答什么。

  比如评估集中有一条数据：

  {
    "question": "Verilog 中 blocking assignment 和 nonblocking assignment 有什么区别？",
    "groundtruth": "阻塞赋值使用 =，按顺序立即执行；非阻塞赋值使用 <=，在当前时间步结束时统一更新，常用于时序逻辑。"
  }

  这里的 groundtruth 就是人工提前写好的参考答案。

  它的作用是：

  - 用来判断模型回答是否正确；
  - 用来算 context_recall，看检索出来的上下文有没有覆盖标准答案所需信息；
  - 用来算回答质量指标，比如答案相关性、事实完整性；
  - 用来做回归测试，看看改了 RAG 或 prompt 后效果有没有变差。

  和模型回答的区别：

  question：用户问题
  groundtruth：人工标准答案
  answer：模型实际回答
  contexts：RAG 检索出来的文档片段

  RAGAS 评估大概就是比较这些东西：

  question + answer + contexts + groundtruth

  举个直观例子：

  问题：setup time violation 怎么优化？

  groundtruth：
  可以通过降低时钟频率、优化组合逻辑、插入流水线、改善布局布线、调整约束等方式优化。

  模型回答：
  可以通过插入流水线、减少组合逻辑路径、降低时钟频率解决。

  检索 contexts：
  包含 setup violation、组合逻辑延迟、流水线优化相关内容。

  如果模型回答和 groundtruth 接近，而且 contexts 也支持这些答案，评估分数就高。

  面试可以这样说：

  > groundtruth 是评估集里的标准答案，通常由人工标注或整理而来。在 RAGAS 评估中，它用于和模型 answer、检索 contexts 一起计算回答相关性、上下文召回率等指标。它
  > 不是模型生成的内容，而是我们用来衡量系统回答质量的参考真值。

```
#### 为什么这是亮点？

因为它评测的不只是 LLM 输出，还覆盖了：

- 检索质量。
- 引用质量。
- 拒答策略。
- 工具选择。

面试表达：

> 我没有只用主观测试判断效果，而是把最终 chat 主链路接入评测。除了 RAGAS 的 faithfulness、answer relevancy 等指标，还补充了 citation correctness、refusal correctness 和 tool routing accuracy，因为这个项目的核心风险不只是答错，还包括引用幻觉和工具路由错误。

---

### 模块九：记忆系统 Memory

关键文件：

- `app/core/memory/manager.py`
- `app/core/memory/short_term.py`
- `app/core/memory/long_term.py`
- `app/core/memory/local.py`
- `app/core/memory/milvus.py`
- `app/core/memory/factory.py`
- `app/api/routes/chat.py`

#### 存在的必要性

没有记忆的问答机器人每一轮都像第一次见用户。用户追问“继续展开”“那 hold 呢？”时，如果系统不记得上一轮，就会答偏。

这个项目的记忆系统解决两个问题：

- 短期记忆：保留最近若干轮 user/assistant 消息，让追问有上下文。
- 长期记忆：把重要内容写入可召回存储，下次同会话提问时可以检索回来。

#### 当前实现

默认实现是本地 JSONL，分成两层：

- `ShortTermMemory`：写入 `data/memory/short_term/*.jsonl`，读取最近 `MEMORY_WINDOW_SIZE` 条消息。
- `LongTermMemory`：写入 `data/memory/long_term/*.jsonl`，按当前 query 和记忆内容的关键词重叠召回 `MEMORY_RECALL_TOP_K` 条。

配置：

```text
MEMORY_ENABLED=true
MEMORY_BACKEND=local
MEMORY_STORE_PATH=data/memory
MEMORY_WINDOW_SIZE=20
MEMORY_RECALL_TOP_K=5
```

可选切换 Milvus 长期记忆：

```text
MEMORY_BACKEND=milvus
MEMORY_MILVUS_COLLECTION_NAME=agent_memory
MEMORY_EMBEDDING_MODEL_PATH=BAAI/bge-m3
```

注意：`MEMORY_BACKEND=milvus` 只替换长期记忆实现；短期记忆仍然固定使用 JSONL。Milvus 不可用时会回退到 JSONL 长期记忆。

核心流程：

```text
chat request
  -> get_memory_manager()
  -> memory.get_context(session_id, latest_user_query)
  -> 注入 short_term_messages 和 long_term_items
  -> Agent 生成回答
  -> memory.save() 保存短期历史
  -> memory.remember() 保存长期记忆
```

#### 面试怎么讲

> 我把记忆拆成短期历史和长期召回。短期历史用 JSONL 保存同一 `conversation_id` 下最近多轮 user/assistant 消息，解决“继续展开”“那 hold 呢？”这类追问；长期记忆也默认用 JSONL 保存关键内容，并通过简单关键词重叠召回，保证本地演示不依赖额外服务。如果配置 `MEMORY_BACKEND=milvus`，只把长期记忆替换成 Milvus 向量召回，短期记忆仍然保持 JSONL。这个取舍适合实习项目：主链路简单可靠，增强能力也有扩展口。

---

### 模块十：强自主 Agent

关键文件：

- `app/core/agent/autonomous.py`
- `app/api/routes/agent.py`
- `app/models/schemas.py`

#### 存在的必要性

普通 `/chat` 更像“问答 Agent”：用户问一次，系统路由工具并回答一次。

`/agent/run` 更像“任务 Agent”：用户给一个目标，系统要能自己拆步骤、执行工具、处理失败、反思质量并输出可复核交付。

#### 核心流程

```text
goal
  -> _plan
  -> execute each AutonomousTaskStep
  -> _invoke_tool_with_audit
  -> _recover on failure
  -> _finalize
  -> _reflect
  -> _build_audit_summary
  -> _derive_status
```

#### 状态设计

`AutonomousTaskStep` 不是只有文本结果，而是包含：

- `rationale`：为什么执行这一步。
- `arguments`：工具参数。
- `observation`：工具或推理观察。
- `evidence`：可审计证据。
- `confidence`：步骤置信度。
- `review_flags`：需要人工复核的原因。
- `error`：工具失败原因。

`AutonomousTask` 额外包含：

- `audit_summary`
- `review_flags`
- `confidence`
- `reflection`
- `final_answer`

#### 面试怎么讲

> 我把强自主 Agent 做成目标驱动的任务闭环，而不是让 LLM 一次性回答。它先规划，再逐步执行工具或推理，每一步都记录为什么做、用了什么参数、拿到了什么证据、置信度如何。如果工具失败，会恢复执行但保留错误；最终状态会根据 error、review_flags、confidence 和 reflection 推导，所以证据不足时是 `needs_review`，不会假装完成。

---

### 模块十一：Web 前端

关键文件：

- `app/static/index.html`
- `app/static/app.js`
- `app/static/styles.css`
- `app/main.py`

#### 存在的必要性

只有 API 的 Agent 不容易演示，也不容易观察工具调用过程。前端让用户可以直接看到：

- 当前会话 ID。
- 聊天内容。
- 工具调用事件。
- 引用来源。
- 自主任务步骤。
- 每一步的置信度、证据数和复核标记。

#### 当前实现

前端是无构建静态页面：

```text
GET /
  -> app/static/index.html
  -> app/static/app.js
  -> app/static/styles.css
```

`app.js` 负责：

- 调用 `/api/v1/chat/stream` 并解析 SSE。
- 维护 localStorage 中的 `conversation_id` 和最近消息。
- 渲染工具事件和引用来源。
- 调用 `/api/v1/agent/run` 并展示任务轨迹。

#### 面试怎么讲

> 我没有把项目停留在接口层，而是加了一个轻量 Web UI。它不是营销页，而是 Agent 工作台：左侧看会话、工具和来源，主区域支持聊天和自主任务。这样演示时能看到 Agent 为什么调用工具、证据来自哪里、哪些结论需要复核。

---

### 模块十二：可靠性审计 Tool Audit

关键文件：

- `app/core/tools/base.py`
- `app/core/tools/registry.py`
- `app/core/tools/builtin/ic_tools.py`
- `tests/test_tool_audit.py`
- `tests/test_autonomous_agent_audit.py`

#### 存在的必要性

强 Agent 最大的问题不是“能不能答”，而是“答得靠谱不靠谱”。如果工具输出只是一段文本，Agent 很难判断证据强弱，也很难告诉用户哪里需要复核。

所以当前项目把工具结果统一成审计结构：

```text
{
  "ok": true,
  "result": "...",
  "summary": "...",
  "evidence": [...],
  "confidence": "high|medium|low|unknown",
  "review_flags": [...]
}
```

#### 核心机制

- `BaseTool.validate_arguments()`：工具执行前校验参数，拒绝缺必填和未知参数。
- `ToolRegistry.invoke_with_audit()`：执行工具并把结果归一化为审计记录。
- `ICRAGSearchTool`：返回检索证据；无结果时标记 `rag_no_results`。
- `VerilogCodeAnalyzerTool`：返回风险 finding、行级 code evidence 和 review flags。
- `TimingConstraintSuggesterTool`：返回 SDC 模板、默认假设和缺失上下文标记。
- `AutonomousAgent._build_audit_summary()`：把步骤级审计汇总成任务级状态。

#### 面试怎么讲

> 我没有只靠 prompt 说“请谨慎回答”，而是在工具层做了硬约束。工具调用前要过参数 schema，工具输出必须带证据、置信度和复核标记。Agent 根据这些字段决定任务是 completed 还是 needs_review。比如 RAG 模型路径坏了，Agent 会恢复执行，但最终会标记工具失败和低置信，而不是输出一个看似确定的答案。

---

## 第三阶段：关键技术点拎出来（面试重点）

### 亮点一：LangGraph 状态图编排领域 Agent

#### 技术本质

把 Agent 拆成显式状态机，而不是单次 LLM 调用。

关键节点：

- `pre_tool_router`
- `tool_executor`
- `answer_generator`
- `clarify`

#### 简历写法

> 基于 LangGraph 设计 IC 垂类 Agent 主链路，实现问题澄清、领域识别、工具路由、RAG 检索和答案生成的状态图编排。

#### 面试怎么讲

> 我把 Agent 的推理流程拆成几个节点。首先 router 判断问题是否太短、是否属于 IC 领域，以及需要哪些工具；然后 tool_executor 调用 RAG、Verilog 分析或时序约束工具；最后 answer_generator 严格基于工具结果回答。这样相比直接调用大模型，更可控、可解释，也方便扩展工具。

#### 面试官可能追问

问题：为什么不用 LangChain Agent 直接 function calling？

回答：

> 这个项目更强调可控流程。LangGraph 可以明确表达状态转移，比如短问题进入 clarify、非 IC 问题进入严格拒答、有工具时执行工具后再生成回答。Function calling 更依赖模型自己决定是否调工具，而我这里希望路由策略可解释、可测试。
```
function calling 是大模型的一种工具调用机制：

  > 模型不只是生成自然语言，还可以根据用户问题，决定调用某个函数/工具，并生成结构化参数。

  比如用户问：

  帮我查一下北京今天的天气

  普通模型可能直接编答案。

  有 function calling 时，你会先把可用工具告诉模型：

  {
    "name": "get_weather",
    "description": "查询城市天气",
    "parameters": {
      "city": "string"
    }
  }

  模型判断需要调用工具后，不直接回答，而是输出类似：

  {
    "tool_name": "get_weather",
    "arguments": {
      "city": "北京"
    }
  }

  然后服务端真正执行函数：

  get_weather(city="北京")

  拿到结果后，再把工具结果交给模型生成最终回答。

  完整流程是：

  用户问题
  -> 模型判断要不要调用工具
  -> 模型生成函数名和参数
  -> 服务端执行真实函数
  -> 工具结果返回给模型
  -> 模型生成最终答案

  它的作用：

  - 让模型能访问外部能力，比如查数据库、查天气、检索知识库、执行计算；
  - 参数是结构化的，比让模型输出一段文本再解析更稳定；
  - 服务端掌控真正执行，模型只是“提出调用意图”；
  - 可以把 LLM 变成 Agent 的决策中心。

  和你这个项目的区别：

  你这个项目目前主要是 规则型工具路由，不是标准 function calling。

  也就是说：

  你的项目：
  pre_tool_router 用代码规则判断调用哪个工具

  function calling：
  让大模型根据工具 schema 判断调用哪个函数

  你项目里工具确实有“工具调用”思想，比如：

  - ic_rag_search
  - verilog_code_analyzer
  - timing_constraint_suggester

  但这些工具不是由模型通过 function calling 自动选择，而是由 pre_tool_router 的规则选择。

  面试可以这样说：

  > function calling 是 LLM 根据开发者提供的函数 schema，自动判断是否需要调用工具，并生成结构化参数的一种机制。服务端拿到函数名和参数后执行真实函数，再把结果交
  > 给模型生成最终回答。我的项目里采用的是规则型工具路由，由 pre_tool_router 根据 IC/Verilog 关键词选择工具，不是完全依赖 LLM function calling。这样成本更低、
  > 行为更可控，但泛化能力不如 LLM 自动工具选择。
```
问题：LangGraph 如果不可用怎么办？

回答：

> 代码里保留了 `_run_without_langgraph` 顺序执行降级，即使缺少 LangGraph 依赖，也能按 pre_tool_router -> tool_executor -> answer_generator 的顺序保持主链路行为一致。

---

### 亮点二：IC 领域 RAG 检索与定制分块

#### 技术本质

面向 IC 文档特点优化 RAG，而不是直接套通用 chunking。

包括：

- IC query expansion。
- Verilog module 边界保留。
- 章节/时序图边界保留。
- Chroma + LlamaIndex 检索。
- CrossEncoder rerank。

#### 简历写法

> 构建 IC 专业文档 RAG 检索链路，基于 LlamaIndex + Chroma + HuggingFace Embedding 实现向量检索，并设计保留 Verilog 模块、章节和时序图边界的领域定制分块器。

#### 面试怎么讲

> IC 文档里有代码、章节、时序图，如果用固定长度切块，可能把 `module ... endmodule` 或时序说明切断，影响检索质量。所以我实现了一个定制 splitter，优先保留 Verilog module、章节标题和 Timing Diagram 边界。检索时再结合 embedding 和 reranker，提高召回和排序质量。

#### 面试官可能追问

问题：chunk size 为什么是 800？

回答：

> 这是一个经验折中。太小会丢上下文，特别是代码块和时序解释；太大则会引入噪声并增加 LLM 上下文成本。当前 800 配合 100 overlap 适合 demo 规模，后续可以通过 retrieval eval 调参。

问题：为什么需要 reranker？

回答：

> 向量检索更偏召回，top-k 里可能有语义相近但不精确的内容。CrossEncoder reranker 会把 query 和候选 chunk 成对输入重新打分，通常能提升 top 结果相关性，尤其适合专业问答场景。

---

### 亮点三：服务端引用治理，降低 RAG 幻觉

#### 技术本质

不要让模型决定最终引用，服务端根据真实检索结果重建引用。

#### 简历写法

> 设计服务端 citation rewriting 机制，移除模型自生成参考资料和未命中伪引用，仅基于本轮真实检索 source/page 生成最终引用列表，降低 RAG 引用幻觉。

#### 面试怎么讲

> RAG 项目里一个常见问题是模型会编造引用。我的处理方式是：prompt 中要求模型不要生成最终参考资料，同时服务端还会强制删除模型自带的参考资料区块，再根据本轮检索结果里的 source/page 重新生成引用列表。这样最终用户看到的引用一定来自真实检索结果。

#### 面试官可能追问

问题：这样能保证回答每句话都有引用吗？

回答：

> 当前能保证最终引用列表来自真实检索结果，但还不是句子级 attribution。更严格的版本可以把回答拆成句子，逐句和 evidence chunk 对齐，没有 evidence 的句子删除或标注无依据。

---

### 亮点四：严格拒答与 out-of-scope 控制

#### 技术本质

垂类 Agent 不应该什么都答。系统通过领域识别、检索命中和严格 prompt 控制回答边界。

#### 简历写法

> 实现 IC Agent 严格模式，对非 IC 问题、检索未命中和证据不足场景进行拒答，避免模型基于通用知识编造答案。

#### 面试怎么讲

> 我没有让模型自由回答所有问题。router 会先判断是否属于 IC 领域；如果不是，就不调用工具并走严格拒答。即使是 IC 问题，如果 RAG 没有命中，answer_generator 也会拒答，明确说明当前知识库未命中，不会用通用知识补答。

#### 面试官可能追问

问题：拒答会不会影响用户体验？

回答：

> 会有取舍。对专业知识库问答来说，可信度比覆盖率更重要。为了改善体验，可以在拒答时给用户建议，比如换问法、上传资料、补充上下文，而不是直接编答案。

---

### 亮点五：工具路由与 IC 专业工具组合

#### 技术本质

把 RAG、代码分析、时序约束建议封装成工具，Agent 根据意图选择。

#### 简历写法

> 封装 IC 专业工具集，包括 RAG 知识检索、Verilog 规则检查和 SDC 时序约束建议，并通过 Agent 路由实现多工具协同。

#### 面试怎么讲

> 用户问普通知识时只需要 RAG；如果问题里包含 Verilog 代码，就可以同时检索知识库并调用代码分析工具；如果是 setup/hold、clock、SDC 相关问题，就可以调用时序约束建议工具。这样系统不是单一问答，而是能根据任务类型调用不同能力。

#### 面试官可能追问

问题：当前路由是规则，会不会不准？

回答：

> 当前规则路由的优势是稳定、可解释，适合 demo 和小规模领域任务。缺点是扩展性一般。后续可以引入意图分类模型，或者让 LLM function calling 给出候选工具，再用规则做安全兜底。

---

### 亮点六：RAGAS + 自定义指标评测闭环

#### 技术本质

用自动化评测验证最终 Agent 链路，而不是只看人工 demo。

#### 简历写法

> 搭建 RAGAS 评测流程，对最终 Chat Agent 链路进行 faithfulness、answer relevancy、context recall/precision 评估，并扩展 citation correctness、refusal correctness、tool routing accuracy 指标。

#### 面试怎么讲

> 我不是只测 retriever，也不是只人工看回答，而是直接调用最终 chat 主链路，拿到 answer、sources 和 tool_events 后统一评测。除了 RAGAS 指标，我还加了引用正确率、拒答正确率和工具路由准确率，因为这些更贴近 Agent 项目的真实风险。

#### 面试官可能追问

问题：RAGAS 有什么局限？

回答：

> RAGAS 本身依赖 judge LLM，会受评测模型质量影响，而且对专业领域问题可能需要更精细的标准答案。所以我会结合人工 spot check、自定义规则指标和固定回归集一起使用。

---

### 亮点七：JSONL 会话记忆与可选 Milvus 长期记忆

#### 技术本质

把 Agent 从无状态 API 升级为有上下文的对话系统。

#### 简历写法

> 设计 Agent 记忆系统，默认支持 JSONL 短期历史窗口和 JSONL 长期关键词召回，并可选切换 Milvus 长期向量召回，实现基于 `conversation_id` 的多轮上下文复用。

#### 面试怎么讲

> 我把记忆拆成两层。短期记忆用 JSONL 保存最近多轮消息，解决用户追问时上下文丢失；长期记忆默认也用 JSONL 保存关键内容，并用简单关键词重叠召回，保证本地可跑、可演示。如果设置 `MEMORY_BACKEND=milvus`，只把长期记忆替换成 embedding + Milvus collection 的向量召回。这样项目不只是单轮问答，而是能围绕一个会话持续积累上下文，同时不会让基础版本依赖太多外部服务。

#### 面试官可能追问

问题：为什么不是只把全部历史塞给模型？

回答：

> 全部历史会增加 token 成本，也会引入噪声。短期窗口保留最近上下文，长期记忆只召回与当前问题相关的片段，能在成本和相关性之间取得平衡。

---

### 亮点八：强自主 Agent 任务闭环

#### 技术本质

把“回答问题”升级为“完成目标”：计划、执行、恢复、反思、审计。

#### 简历写法

> 实现强自主 Agent 模式，支持目标规划、多步骤工具执行、失败恢复、反思审查和任务级审计摘要，并通过 `needs_review` 状态控制低证据输出风险。

#### 面试怎么讲

> 普通 `/chat` 是问答链路，`/agent/run` 是任务链路。用户给一个目标后，Agent 先生成计划，再逐步执行工具或推理。每一步都会记录 rationale、arguments、observation、evidence、confidence 和 review_flags。如果工具失败，它不会直接崩掉，而是降级恢复，同时保留错误。最终根据证据和反思结果推导状态，所以低证据任务会返回 `needs_review`。

#### 面试官可能追问

问题：这算强自主 Agent 吗？

回答：

> 它已经具备强自主 Agent 的核心形态：目标驱动、自动规划、多步执行、工具使用、失败恢复、自我审查和状态判断。但它还不是完全生产级 autonomic system，因为现在任务是同步执行，工具集也有限，后续可以加后台任务队列、更多工程工具和人工审批节点。

---

### 亮点九：工具审计与可靠性控制

#### 技术本质

把“工具返回一段文本”升级为“工具返回可审计证据结构”。

#### 简历写法

> 设计工具审计层，对工具参数进行 schema 校验，并将工具输出统一归一化为 evidence、confidence、review_flags 和 summary，支持 Agent 自动判断结果是否需要人工复核。

#### 面试怎么讲

> 我没有只靠 prompt 要求模型谨慎，而是在工具层做硬约束。工具执行前先校验参数，执行后统一输出证据、置信度和复核标记。比如 RAG 没有检索结果会标记 `rag_no_results`，SDC 缺少 IO 背景会标记 `missing_input_delay_context`，Verilog 规则命中会返回行级证据。自主 Agent 根据这些字段决定任务状态，而不是盲目相信模型总结。

#### 面试官可能追问

问题：为什么 confidence 用 high/medium/low，而不是一个精确分数？

回答：

> 当前工具以规则型和检索型为主，很多结果没有严格概率含义。用 high/medium/low 更诚实，也更适合前端展示和任务状态判断。后续如果引入 calibrated evaluator，可以再扩展为数值分数。

---

## 第四阶段：我必须掌握的知识清单

### 1. HTTP 与 FastAPI

#### 为什么项目用到了它

项目通过 FastAPI 对外提供服务，包括：

- 非流式 chat。
- SSE 流式 chat。
- 强自主 Agent 任务。
- 文档上传。
- 健康检查。
- 静态前端页面。

#### 不会会被问死在哪里

面试官可能问：

- POST `/chat` 的请求和响应是什么？
- SSE 和普通 HTTP 响应有什么区别？
- `/agent/run` 和 `/chat` 有什么区别？
- 为什么 AI 应用常需要 streaming？
- FastAPI 的 async 有什么意义？

你需要掌握：

- HTTP method、status code、JSON body。
- Pydantic request/response model。
- SSE 基本格式：`event: xxx` + `data: xxx`。
- 异步 I/O 和 LLM 调用等待之间的关系。

---

### 2. Python 异步编程

#### 为什么项目用到了它

- FastAPI endpoint 是 async。
- Agent `run` 是 async。
- 工具调用可能包含 IO 或模型调用。
- 检索里用 `asyncio.to_thread` 包装同步检索。

#### 不会会被问死在哪里

面试官可能问：

- `async def` 和普通函数区别是什么？
- `await agent.run()` 在等什么？
- 为什么同步 CPU/阻塞任务要放到 thread？

你需要掌握：

- coroutine。
- event loop。
- await。
- 阻塞调用对 async server 的影响。

---

### 3. RAG 基础

#### 为什么项目用到了它

项目核心能力是基于本地 IC 文档回答问题。

RAG 基本流程：

```text
文档解析 -> 分块 -> embedding -> 向量库 -> 检索 -> rerank -> 拼上下文 -> LLM 生成
```

#### 不会会被问死在哪里

面试官可能问：

- 为什么不用大模型直接回答？
- embedding 是什么？
- chunk size 怎么影响效果？
- reranker 和 retriever 区别是什么？
- RAG 怎么降低幻觉？

你需要掌握：

- dense retrieval。
- vector database。
- top-k。
- chunk overlap。
- context precision / recall。
- faithfulness。

---

### 4. 向量数据库 Chroma

#### 为什么项目用到了它

Chroma 用来持久化文档 chunk 的向量和 metadata。

#### 不会会被问死在哪里

面试官可能问：

- 向量库里存的是什么？
- metadata 有什么用？
- 为什么 source/page 要存 metadata？
- data 文件变了，向量库没变怎么办？

你需要掌握：

- collection。
- vector embedding。
- metadata filtering。
- persistent storage。
- source consistency check。

---

### 5. LangGraph / Agent 编排

#### 为什么项目用到了它

项目需要显式控制：

- 是否澄清。
- 是否调用工具。
- 调哪些工具。
- 什么时候拒答。
- 什么时候生成最终回答。

#### 不会会被问死在哪里

面试官可能问：

- Agent 和普通 chatbot 区别是什么？
- LangGraph 的 state 是什么？
- 节点之间怎么跳转？
- 为什么要工具路由？

你需要掌握：

- state graph。
- node。
- edge。
- conditional edge。
- tool execution。
- ReAct / Plan-and-Execute 基本思想。

---

### 6. LLM Prompt 与严格模式

#### 为什么项目用到了它

最终回答仍然由 LLM 生成，但必须限制它：

- 只能基于工具结果。
- 不能编造引用。
- 未命中时拒答。
- 不要自己输出最终参考资料。

#### 不会会被问死在哪里

面试官可能问：

- prompt 能完全防止幻觉吗？
- 为什么还要服务端引用重写？
- 严格拒答怎么实现？

你需要掌握：

- system prompt。
- grounding。
- hallucination。
- refusal。
- post-processing guardrail。

---

### 7. Verilog / IC 基础

#### 为什么项目用到了它

这是项目的领域基础。没有 IC 基础，很难讲清为什么工具和分块要这样设计。

你需要掌握：

- Verilog module / endmodule。
- always / assign。
- blocking `=` 和 non-blocking `<=`。
- 组合逻辑和时序逻辑。
- latch 风险。
- setup / hold。
- SDC create_clock、input_delay、output_delay。
- 关键路径、流水线、乘法器结构。

#### 不会会被问死在哪里

面试官可能问：

- 为什么 Verilog module 不能随便切断？
- always 块有什么常见问题？
- setup/hold 是什么？
- SDC 约束是给谁用的？
- 乘法器优化有哪些方向？

---

### 8. 评测与指标

#### 为什么项目用到了它

项目需要证明效果，而不是只靠主观 demo。

#### 不会会被问死在哪里

面试官可能问：

- 怎么评估 RAG 系统？
- faithfulness 是什么？
- context precision 和 context recall 区别是什么？
- tool routing accuracy 怎么算？
- 为什么要评估 refusal correctness？

你需要掌握：

- RAGAS 基本指标。
- golden dataset。
- regression test。
- 自动评测和人工评测的边界。

---

## 面试高频问答准备

### Q1：这个项目和普通 ChatGPT 套壳有什么区别？

回答：

> 普通 ChatGPT 套壳是用户问题直接给模型，回答依赖模型自身知识。我的项目是一个垂类 Agent 系统，先通过 LangGraph 做意图路由，再调用 IC RAG 检索、Verilog 分析、时序约束建议等工具，最后基于工具结果回答。并且我做了服务端 citation rewriting，只展示真实检索到的 source/page，还加入了 RAGAS 和自定义指标评测，所以它更强调专业性、可控性和可验证性。

新版补充：

> 现在项目还加入了 JSONL 会话记忆、可选 Milvus 长期记忆、Web UI 和强自主 Agent。强自主模式不是单轮回答，而是会自动规划、执行工具、失败恢复、反思审查，并返回 evidence、confidence 和 review_flags，所以更接近一个可审计的任务型 Agent。

---

### Q2：项目里最有技术含量的地方是什么？

回答：

> 我认为有五个点。第一是 LangGraph 主链路，把澄清、工具路由、工具执行、答案生成拆成显式状态图。第二是 IC 定制 RAG，不只是普通向量检索，还做了 Verilog module、章节和时序图边界保留，以及 query expansion 和 rerank。第三是引用治理，服务端会删除模型自造参考资料，只根据真实检索结果重写引用。第四是强自主 Agent，支持计划、执行、恢复、反思和审计。第五是工具可靠性审计，工具输出会带 evidence、confidence 和 review_flags。

---

### Q3：如果让你继续优化这个项目，你会做什么？

回答：

> 我会从四方面优化。第一是检索质量，引入 hybrid search、metadata filtering 和更细的 chunk hierarchy。第二是工具能力，Verilog 分析接入 Verilator 或 yosys，从正则检查升级到真实语法/综合检查。第三是自主任务工程化，把同步 `/agent/run` 升级成后台队列、任务取消和人工审批。第四是观测与评测，把 trace 细化到每个节点，并扩展工具审计指标。

---

### Q4：你怎么证明这个项目效果好？

回答：

> 项目里有 evaluation 脚本，会读取固定测试集，直接调用最终 chat 主链路，而不是只测单独 retriever。评测指标包括 RAGAS 的 faithfulness、answer relevancy、context recall、context precision，同时我还补充了 citation correctness、refusal correctness 和 tool routing accuracy。新增的工具审计测试还覆盖了参数校验、Verilog 风险标记、SDC 缺失上下文标记，以及自主 Agent 在 RAG 无证据时必须返回 needs_review。

---

### Q5：RAG 仍然可能答错，你怎么降低风险？

回答：

> 我从六层降低风险。第一，router 限制问题范围，非 IC 问题严格拒答。第二，检索未命中时不让模型用通用知识补答。第三，prompt 要求回答必须基于工具结果。第四，服务端 citation rewriter 会移除模型自造引用，只展示本轮真实检索到的 source/page。第五，工具层输出 evidence、confidence 和 review_flags。第六，自主 Agent 根据这些审计字段把低证据任务标记为 needs_review。后续还可以做句子级 evidence 对齐。

---

### Q6：为什么做短问题澄清？

回答：

> 像“乘法器”“时序”这种问题太泛，直接回答容易变成长篇泛泛解释，也可能检索目标不明确。项目里对过短 query 会先澄清，引导用户补充场景，比如“乘法器时序优化有哪些方法”或“SDC 里 setup/hold 约束怎么写”。这能提升回答相关性，也减少无意义长答。

---

### Q7：你这个现在是 Agent 还是问答机器人？

回答：

> 两种模式都有。`/chat` 是带工具调用和 RAG 的问答 Agent，重点是单轮或多轮专业问答；`/agent/run` 是更强的任务型 Agent，重点是给定目标后自动规划、执行、恢复、反思和审计。所以它已经不是普通问答机器人，而是一个带记忆、工具和自主任务闭环的 IC 领域 Agent。

---

### Q8：你怎么让强自主 Agent 更靠谱？

回答：

> 我没有只靠模型自己判断，而是在工具和任务层加了审计机制。工具调用前做参数校验，工具输出统一包含 evidence、confidence、review_flags。自主 Agent 每一步都记录 rationale、arguments、observation 和 error；如果工具失败或证据不足，任务会进入 needs_review，而不是伪装成 completed。前端也会把这些复核原因展示出来。

---

## 你应该能画出来的架构图

面试白板版：

```text
                    +----------------------+
                    | User / Browser / API |
                    +----------+-----------+
                               |
                               v
                    +----------+-----------+
                    | FastAPI API Layer    |
                    | / /chat /stream      |
                    | /agent/run /document |
                    +----------+-----------+
                               |
          +--------------------+--------------------+
          |                                         |
          v                                         v
+---------+-----------+                 +-----------+----------+
| LangGraphICAgent    |                 | AutonomousAgent      |
| router -> tools     |                 | plan -> execute      |
| -> answer           |                 | recover -> reflect   |
+---------+-----------+                 +-----------+----------+
          |                                         |
          +--------------------+--------------------+
                               |
                               v
                    +----------+-----------+
                    | ToolRegistry Audit   |
                    | schema / evidence    |
                    | confidence / review  |
                    +----------+-----------+
                               |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
+-------+------+      +--------+--------+      +------+-------+
| IC RAG Search |      | Verilog Analyzer|      | SDC Suggest |
+-------+------+      +-----------------+      +------+-------+
        |
        v
+-------+----------------------------------------------+
| LlamaIndex + Chroma + Embedding + Reranker            |
| IC Splitter + citation_rewriter                       |
+-------+----------------------------------------------+
        |
        v
+-------+----------------------------------------------+
| Memory: JSONL short-term + JSONL keyword long-term    |
|         optional Milvus long-term recall              |
+------------------------------------------------------+
```

---

## 你应该熟悉的核心文件顺序

建议按这个顺序读代码：

1. `README.md`：理解项目定位和主链路。
2. `app/main.py`：理解 FastAPI 如何启动和挂载路由。
3. `app/static/index.html`、`app/static/app.js`、`app/static/styles.css`：理解前端如何调用 chat 和 agent。
4. `app/api/routes/chat.py`：理解普通问答如何读取记忆、进入 Agent、保存记忆。
5. `app/core/agent/langgraph_agent.py`：理解普通问答 Agent 状态图。
6. `app/api/routes/agent.py`：理解强自主任务入口。
7. `app/core/agent/autonomous.py`：理解计划、执行、恢复、反思和审计。
8. `app/core/tools/base.py`、`app/core/tools/registry.py`：理解工具参数校验和审计归一化。
9. `app/core/tools/builtin/ic_tools.py`：理解三个 IC 工具。
10. `app/core/memory/manager.py`、`app/core/memory/short_term.py`、`app/core/memory/long_term.py`、`app/core/memory/milvus.py`：理解记忆系统。
11. `app/core/rag/retriever.py`：理解 RAG 检索和索引维护。
12. `app/etl/ic_text_splitter.py`：理解 IC 定制分块。
13. `app/core/rag/citation_rewriter.py`：理解引用治理。
14. `tests/test_memory.py`、`tests/test_tool_audit.py`、`tests/test_autonomous_agent_audit.py`：理解新增功能回归测试。
15. `evaluation/evaluate_ragas.py`：理解评测闭环。

---

## 最终项目表达模板

如果面试官让你介绍项目，可以直接按这个模板说：

> 我做的是一个面向集成电路领域的 AI Agent 服务，主要用于 IC 专业知识问答和工程辅助。系统后端用 FastAPI 提供 HTTP、SSE 和自主任务接口，前端提供一个轻量 Agent 工作台。普通问答链路用 LangGraph 编排，包括记忆读取、问题澄清、工具路由、工具执行和答案生成；知识类问题会调用 LlamaIndex + Chroma 的 RAG 检索链路，文档进入知识库前会经过 IC 定制分块，尽量保留 Verilog module、章节和时序图边界；Verilog 代码问题会调用规则型代码分析工具；时序问题会生成 SDC 约束建议。为了降低幻觉，我做了严格拒答、服务端引用重写和工具审计，最终引用只来自本轮真实检索到的 source/page，工具结果也会返回 evidence、confidence 和 review_flags。除此之外，项目支持 JSONL 短期历史和 JSONL 长期关键词召回，并保留 Milvus 长期向量召回作为可选增强；同时新增 `/agent/run` 强自主 Agent，可以对目标自动规划、执行、失败恢复、反思审查和输出 audit_summary。最后，项目用 RAGAS、自定义指标和新增测试覆盖 answer quality、citation correctness、refusal correctness、tool routing accuracy、记忆和工具审计，形成完整质量闭环。

---

## 一句话总结

这个项目最核心的价值不是“调用了大模型”，而是把大模型放进了一个可控、可追溯、可复核的 IC 专业 Agent 工程体系里：有领域路由、有工具调用、有检索增强、有记忆系统、有自主任务闭环、有引用治理、有严格拒答、有可靠性审计、有前端展示，也有评测闭环。
