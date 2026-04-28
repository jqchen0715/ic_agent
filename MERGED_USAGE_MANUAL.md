# 合并后项目使用说明书（project-python）

这份说明书用于解决“两个项目合并后结构和运行方式不清晰”的问题。  
目标是让你按本文一步一步执行，就能完成：

1. 环境准备
2. 服务启动
3. 文档入库
4. 对话调用（非流式/流式）
5. 排障定位

---

## 1. 先明确：哪个仓库是最终仓库

- 最终展示仓库：`project-python`
- 旧仓库（能力来源）：`IC-Expert-agent`
- 迁移思路：保留 `project-python` 的工程分层，把 `IC-Expert-agent` 的 IC 垂类能力迁入

当前主链路已经是：

`/api/v1/chat -> LangGraph Agent(pre_tool_router -> tool_executor -> answer_generator)`

---

## 2. 合并后的目录怎么理解

```text
project-python/
├── app/
│   ├── api/                    # HTTP/SSE 接口层
│   │   └── routes/
│   │       ├── chat.py         # /chat, /chat/stream
│   │       ├── document.py     # /documents/upload, /documents
│   │       └── health.py       # /health, /health/ready
│   ├── core/
│   │   ├── agent/
│   │   │   └── langgraph_agent.py    # 主编排、工具路由、严格拒答
│   │   ├── tools/
│   │   │   ├── registry.py            # 工具注册中心
│   │   │   └── builtin/ic_tools.py    # 3个 IC 工具
│   │   └── rag/
│   │       ├── retriever.py           # Chroma + LlamaIndex 检索
│   │       └── citation_rewriter.py   # 服务端引用重写
│   ├── etl/
│   │   ├── ic_text_splitter.py        # IC 定制分块器（迁移自旧仓库）
│   │   └── pipeline.py                # 解析->分块流水线
│   ├── infrastructure/
│   │   └── database/                  # DB engine/session/model
│   ├── models/schemas.py              # API Schema
│   └── main.py                        # FastAPI 入口
├── data/                              # PDF 知识库目录（检索源）
├── chroma_db/                         # Chroma 持久化目录
├── evaluation/
│   └── retrieval_smoke_test.py        # 检索冒烟测试
└── MERGED_USAGE_MANUAL.md             # 本说明书
```

---

## 3. 运行前准备

## 3.1 Python 与依赖

- 推荐 Python：`3.11+`
- 在 `project-python` 根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 3.2 环境变量

先复制：

```bash
cp .env.example .env
```

最少需要确认这些值：

```env
OPENAI_API_KEY=你的key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/agent_db

# RAG 相关（可不填，默认就是这些）
DATA_PATH=data
CHROMA_PATH=chroma_db
CHROMA_COLLECTION_NAME=ic_expert
EMBEDDING_MODEL_PATH=BAAI/bge-m3
EMBEDDING_DEVICE=cpu
SOURCE_MISMATCH_STRATEGY=rebuild
```

说明：

- `OPENAI_API_KEY` 未配置时，`/chat` 会返回 503
- 首次检索会拉取 embedding 模型（可能较慢）

## 3.3 PostgreSQL 准备

`document` 路由依赖 PostgreSQL。  
你可以二选一：

1. 本机已有 PostgreSQL（并创建 `agent_db`）
2. 用 docker-compose 启动依赖

```bash
docker compose up -d postgres redis
```

---

## 4. 一次性初始化数据库表

当前仓库没有内置 migration 脚本，第一次建议手动建表。

```bash
python3 - <<'PY'
import os
import asyncio
from app.infrastructure.database.models import Base
from app.infrastructure.database.session import init_engine

async def main():
    engine = init_engine(os.getenv("DATABASE_URL"))
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

asyncio.run(main())
PY
```

---

## 5. 启动服务

在 `project-python` 根目录：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

快速检查：

```bash
curl http://127.0.0.1:8000/api/v1/health
```

---

## 6. 使用流程（推荐）

1. 准备 PDF 放入 `data/`（或走上传接口）
2. 调 `/api/v1/chat` 或 `/api/v1/chat/stream` 提问
3. 检查响应里的 `sources` 与“参考资料（服务端生成）”

---

## 7. API 使用示例

## 7.1 非流式对话

接口：`POST /api/v1/chat`

```bash
curl -s http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "乘法器时序优化有哪些方法？"}
    ],
    "temperature": 0.2
  }'
```

返回重点字段：

- `answer`：最终答案
- `content`：兼容字段（与 answer 相同）
- `trace_id`
- `sources[]`：结构化检索结果（`content/source/page/score/chunk_id`）
- `tool_events[]`

## 7.2 流式对话（SSE）

接口：`POST /api/v1/chat/stream`

```bash
curl -N http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "乘法器时序优化有哪些方法？"}
    ]
  }'
```

固定事件类型：

- `tool_call`
- `tool_result`
- `answer`
- `citation`
- `done`
- `error`

## 7.3 文档上传

接口：`POST /api/v1/documents/upload`

```bash
curl -s http://127.0.0.1:8000/api/v1/documents/upload \
  -F "file=@/absolute/path/your_ic_doc.pdf"
```

上传后行为：

1. ETL 解析 + IC 定制分块
2. PDF 同步到 `data/`
3. 自动重建 Chroma 索引（保证可检索）
4. 文档/分块写入 PostgreSQL

---

## 8. Agent 路由规则与回答策略

## 8.1 工具路由

- IC 知识问题 -> `ic_rag_search`
- 包含 `module/always/assign/Verilog/RTL` -> `verilog_code_analyzer`
- 包含 `SDC/时序/clock/setup/hold/false path` -> `timing_constraint_suggester`
- 过短问题（如“乘法器”“时序”）-> 先澄清

## 8.2 严格拒答

- RAG 未命中时走严格拒答路径
- 不允许模型脱离知识库自由补答

## 8.3 引用重写

- 服务端会重写“参考资料”
- 只允许本轮真实检索到的 `source/page`
- 模型伪引用会被移除并提示

---

## 9. 检索冒烟测试（独立脚本）

```bash
python3 evaluation/retrieval_smoke_test.py \
  --query "乘法器时序优化有哪些方法？" \
  --top-k 3
```

预期：

- 至少输出 1-3 条结果
- 每条包含 `source/page`

---

## 10. 常见问题排查

## 10.1 `ModuleNotFoundError: chromadb` 或 `pydantic`

说明依赖没装完整，重新执行：

```bash
pip install -r requirements.txt
```

## 10.2 `OPENAI_API_KEY` 未配置

- `/chat` 会返回 503
- 检查 `.env` 中 `OPENAI_API_KEY`

## 10.3 上传时报数据库表不存在

- 先执行第 4 节的一次性建表脚本

## 10.4 上传后检索慢

- 现在策略是“上传 PDF 后重建索引”，首次或大文档会慢
- 这是当前设计预期，不是报错

## 10.5 `source` 不一致提示

- 系统会比较 `data/*.pdf` 与 Chroma `source`
- `SOURCE_MISMATCH_STRATEGY=rebuild` 时自动重建
- `warn` 时只告警不重建

---

## 11. 从旧仓库迁入了什么（便于你讲故事）

- `IC-Expert-agent/src/ic_text_splitter.py` -> `app/etl/ic_text_splitter.py`
- `IC-Expert-agent/src/llama_index_rag.py` -> `app/core/rag/retriever.py`
- `IC-Expert-agent/src/rag_core.py` 三工具 + 路由规则 + 严格拒答思想 -> `app/core/tools` + `app/core/agent/langgraph_agent.py`
- `IC-Expert-agent/src/server.py` 引用治理思路 -> `app/core/rag/citation_rewriter.py`

---

## 12. 一条龙最小演示清单

1. 启动依赖（Postgres）
2. 安装 Python 依赖
3. 配置 `.env`
4. 建表
5. 启动 FastAPI
6. 上传一个 IC PDF
7. 调 `/api/v1/chat` 提问“乘法器时序优化有哪些方法？”
8. 查看 `sources` 和“参考资料（服务端生成）”

## 使用
project-python 里已经有“知识库切割/分块”程序，只是名字不是“切割.py”。
你可以看这几个文件：

分块器实现（IC 定制规则）
ic_text_splitter.py (line 12)

分块策略入口（IC_CUSTOM）
chunker.py (line 19)
chunker.py (line 64)

上传接口已实际使用 IC 分块策略
document.py (line 82)

RAG 建索引时也用同一个 IC 分块器
retriever.py (line 127)

平时问答走“增量复用”；只有“索引缺失/不一致”或“你上传了新 PDF”才会重建。