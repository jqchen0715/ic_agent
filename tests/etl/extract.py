import json
import random
from pathlib import Path
import sys

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings


VERILOG_KEYWORDS = ("module", "always", "assign", "verilog", "rtl")
TIMING_KEYWORDS = ("sdc", "时序", "clock", "setup", "hold", "false path", "false_path")
KNOWLEDGE_CUES = (
    "什么",
    "为何",
    "为什么",
    "怎么",
    "如何",
    "区别",
    "原理",
    "方法",
    "有哪些",
    "优化",
    "what",
    "why",
    "how",
)


def infer_expected_tools(question: str) -> list[str]:
    q_lower = question.lower()
    expected_tools: list[str] = []

    is_verilog = any(k in q_lower for k in VERILOG_KEYWORDS)
    is_timing = any(k in q_lower for k in TIMING_KEYWORDS)
    is_knowledge = any(k in question or k in q_lower for k in KNOWLEDGE_CUES)

    if is_knowledge:
        expected_tools.append("ic_rag_search")
    if is_verilog:
        expected_tools.append("verilog_code_analyzer")
    if is_timing:
        expected_tools.append("timing_constraint_suggester")

    if not expected_tools:
        expected_tools.append("ic_rag_search")

    return expected_tools


settings = get_settings()


def project_path(*parts):
    return PROJECT_ROOT.joinpath(*parts)


embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model_path,
    model_kwargs={"device": settings.embedding_device},
)


def _build_llm() -> ChatOpenAI:
    common_kwargs = {
        "temperature": 0.0,
        "api_key": settings.openai_api_key,
        "base_url": settings.openai_api_base or None,
    }
    # 兼容不同版本 langchain_openai 的参数名差异（model vs model_name）。
    try:
        return ChatOpenAI(model=settings.openai_model, **common_kwargs)
    except TypeError:
        return ChatOpenAI(model_name=settings.openai_model, **common_kwargs)


llm = _build_llm()


def generate_eval_dataset(sample_size=30):
    print("正在连接本地 ChromaDB 数据库...")

    # 2. 我们在这里自己连一下数据库（解决找不到 db 的问题）

    db = Chroma(
    persist_directory=str(project_path("chroma_db")),
    embedding_function=embeddings,
    collection_name=settings.chroma_collection_name,  # 关键
)

    # 3. 穿透到底层，获取 ChromaDB 里的所有真实数据
    collection = db._collection
    all_data = collection.get()
    all_documents = all_data['documents']
    all_metadatas = all_data['metadatas']

    total_docs = len(all_documents)
    print(f"数据库中共有 {total_docs} 个文档块。")

    if total_docs == 0:
        print("数据库为空，请先运行 create_vector_db.py 构建知识库。")
        return

    # 4. 随机抽取候选样本（给一些冗余，避免某些条目生成失败）
    max_attempts = min(total_docs, sample_size * 4)
    indices = random.sample(range(total_docs), max_attempts)
    dataset = []
    output_file = project_path("evaluation", "datasets", "eval_dataset_qg_30.json")

    # 5. 定义大模型的出题 Prompt
    prompt = PromptTemplate.from_template("""
你是集成电路领域评测集构建助手。请基于给定参考文本，生成 1 条可用于 RAG 评测的数据。

要求：
1. 问题必须能直接从参考文本回答，避免开放式提问。
2. ground_truth 必须是简洁、准确、可核验的标准答案。
3. 不要编造参考文本中没有的信息。
4. expected_tools 必须根据问题意图设置：IC 知识检索用 ic_rag_search；Verilog/RTL 代码分析用 verilog_code_analyzer；SDC/时序约束建议用 timing_constraint_suggester。
5. 严格输出 JSON，不要任何额外文字，格式如下：
{{
  "question": "...",
  "ground_truth": "...",
  "expected_tools": ["ic_rag_search"]
}}

参考文本：
{text}
""")

    print("\n开始生成 question + ground_truth 评测集，请稍候...\n")

    # 6. 遍历抽取的真实文本，让大模型自动生成问题
    for i, idx in enumerate(indices):
        if len(dataset) >= sample_size:
            break

        real_text = all_documents[idx]
        source_file = all_metadatas[idx].get('source', 'unknown.pdf') if all_metadatas[idx] else 'unknown.pdf'

        if len(real_text.strip()) < 50:
            continue

        try:
            chain = prompt | llm
            response = chain.invoke({"text": real_text[:1800]})

            result_str = response.content.strip().replace("```json", "").replace("```", "")
            generated_data = json.loads(result_str)

            question = generated_data["question"].strip()
            expected_tools = infer_expected_tools(question)

            # 仅保留评测必需字段；额外保留 source 便于人工抽查
            dataset_item = {
                "question": question,
                "ground_truth": generated_data["ground_truth"].strip(),
                "expected_tools": expected_tools,
                "source": source_file,
            }
            dataset.append(dataset_item)

            # 增量写盘：即使中断也能保留已完成数据
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)

            print(f"✅ 成功生成第 {len(dataset)} 条: {dataset_item['question']}")

        except Exception as e:
            print(f"⚠️ 第 {i + 1} 条生成失败，跳过。错误信息: {e}")
            continue

    # 7. 保存文件
    # 7. 最终落盘
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 自动提取完成！共生成 {len(dataset)} 条评测数据。")
    print(f"文件已保存为: {output_file}")
    return dataset


if __name__ == "__main__":
    generate_eval_dataset(30)
