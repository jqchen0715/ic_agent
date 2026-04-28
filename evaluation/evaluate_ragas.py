# -*- coding: utf-8 -*-
"""RAGAS 评测入口：打最终 Chat 主链路，输出 summary/details。"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from pathlib import Path
import sys
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from app.api.routes.chat import _build_ic_agent
from app.config import get_settings
from app.core.rag.citation_rewriter import rewrite_answer_citations

TEST_CASES_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "eval_dataset_qg_30.json"
REPORT_DIR = PROJECT_ROOT / "evaluation" / "reports"

settings = get_settings()
judge_llm = ChatOpenAI(
    model=settings.openai_model,
    temperature=0.0,
    api_key=settings.openai_api_key,
    base_url=settings.openai_api_base or None,
)
ragas_llm = LangchainLLMWrapper(judge_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name=settings.embedding_model_path,
        model_kwargs={"device": settings.embedding_device},
    )
)
# 在线模型评测容易受网络抖动影响；适当放宽超时并控制并发。
run_config = RunConfig(timeout=300, max_workers=1)


def _ensure_eval_llm_credentials():
    if not settings.openai_api_key:
        raise RuntimeError(
            "评测失败：缺少 OPENAI_API_KEY。请在 .env 或当前环境中配置 OpenAI-compatible API Key。"
        )
    if not settings.openai_api_base:
        raise RuntimeError(
            "评测失败：缺少 OPENAI_API_BASE。"
            "例如：https://dashscope.aliyuncs.com/compatible-mode/v1"
        )


def _normalize_source_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    normalized = name.strip().replace("\\", "/")
    if "/" in normalized:
        normalized = normalized.split("/")[-1]
    return normalized.lower()


def _source_value(source: Any) -> str:
    if isinstance(source, dict):
        return str(source.get("source") or source.get("file_name") or "")
    metadata = getattr(source, "metadata", {}) or {}
    return str(metadata.get("source") or metadata.get("file_name") or "")


def _extract_retrieved_sources(sources: list[Any]) -> set[str]:
    out = set()
    for item in sources or []:
        source = _normalize_source_name(_source_value(item))
        if source:
            out.add(source)
    return out


def _extract_cited_sources(answer: str) -> list[str]:
    text = answer or ""
    cited = []

    in_server_reference_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "参考资料（服务端生成）":
            in_server_reference_block = True
            continue
        if not in_server_reference_block:
            continue

        m = re.match(r"^\d+\.\s*(?P<source>[^|\n]+?)\s*\|\s*(?P<page>.+?)\s*$", stripped)
        if m:
            candidate = _normalize_source_name(m.group("source"))
            if candidate:
                cited.append(candidate)
            continue
        if stripped and not re.match(r"^\d+\.", stripped):
            break

    # 兼容旧格式：
    # [R1] xxx.pdf，第12页
    # [R2] 书名, 第4.5节
    pattern = re.compile(r"\[R\d+\]\s*([^\n，,|]+)")
    for m in pattern.finditer(text):
        candidate = _normalize_source_name(m.group(1))
        if candidate:
            cited.append(candidate)

    # 去重并保持顺序
    seen = set()
    uniq = []
    for c in cited:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _is_strict_reject_answer(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return False

    required_any = [
        "【严格拒答】",
        "当前知识库未命中",
        "无法基于资料直接回答",
        "知识库中未找到相关信息",
        "当前为严格模式",
        "不会基于通用知识自行补答",
    ]
    return any(k in text for k in required_any)


def _is_expected_refusal_case(case: dict[str, Any]) -> bool:
    expected_tools = _get_case_value(case, "expected_tools", "expected_tool", default=None)
    ground_truth = str(_get_case_value(case, "ground_truth", "ground_truths", "reference", "answer")).strip()
    source = str(_get_case_value(case, "source", "expected_source", default="")).strip().lower()
    no_expected_tools = expected_tools == [] or expected_tools == "[]"
    return no_expected_tools or _is_strict_reject_answer(ground_truth) or source in {"n/a", "na", "none", "无"}


def load_test_cases(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到测试集文件: {path}\n"
            "请创建 JSON 列表，每条至少包含 question 和 ground_truth；"
            "可选字段 expected_tools 用于计算 tool_routing_accuracy。"
        )
    with path.open("r", encoding="utf-8") as f:
        test_cases = json.load(f)
    if not isinstance(test_cases, list):
        raise ValueError("测试集格式错误：应为列表(list)")
    return test_cases


def _get_case_value(case: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in case and case[key] is not None:
            return case[key]
    return default


def _source_content(source: Any) -> str:
    if isinstance(source, dict):
        return str(source.get("content") or source.get("snippet") or "")
    return str(getattr(source, "page_content", "") or "")


def _selected_tools(tool_events: list[dict[str, Any]]) -> list[str]:
    seen = set()
    out = []
    for event in tool_events or []:
        tool = str(event.get("tool", "")).strip()
        if tool and tool not in seen:
            seen.add(tool)
            out.append(tool)
    return out


def _tool_routing_score(expected: Any, actual: list[str]) -> float | None:
    if expected in (None, ""):
        return None
    if isinstance(expected, str):
        expected_tools = {expected}
    else:
        expected_tools = {str(x) for x in expected or [] if str(x).strip()}
    if not expected_tools:
        return None
    return 1.0 if expected_tools.issubset(set(actual)) else 0.0


async def _run_final_chat_chain(question: str, agent: Any | None = None) -> dict[str, Any]:
    agent = agent or _build_ic_agent()
    result = await agent.run(
        messages=[{"role": "user", "content": question}],
        model_preference=settings.openai_model,
        temperature=0.0,
    )
    rewritten = rewrite_answer_citations(result.content, result.sources)
    return {
        "answer": rewritten.answer,
        "sources": result.sources,
        "tool_events": result.tool_events,
        "model": result.model_id or settings.openai_model,
    }


async def _run_final_chat_chain_batch(questions: list[str]) -> list[dict[str, Any]]:
    agent = _build_ic_agent()
    results = []
    for question in questions:
        results.append(await _run_final_chat_chain(question, agent))
    return results


def _mean_or_none(values: list[float | None]) -> float | None:
    numeric = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(pd.Series(numeric, dtype="float").mean()) if numeric else None


def run_ragas_evaluation():
    _ensure_eval_llm_credentials()
    test_cases = load_test_cases(TEST_CASES_PATH)
    print("🔄 正在打最终 Chat 主链路并评估...")
    data = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}
    per_case_stats = []

    prepared_cases = []
    questions = []
    for case in test_cases:
        question = str(_get_case_value(case, "question", "query")).strip()
        ground_truth = str(_get_case_value(case, "ground_truth", "ground_truths", "reference", "answer")).strip()
        expected_tools = _get_case_value(case, "expected_tools", "expected_tool", default=None)
        if not question or not ground_truth:
            raise ValueError(f"测试集样本缺少 question/ground_truth: {case}")
        is_refusal_case = _is_expected_refusal_case(case)
        prepared_cases.append((question, ground_truth, expected_tools, is_refusal_case))
        questions.append(question)

    chat_results = asyncio.run(_run_final_chat_chain_batch(questions))

    for (question, ground_truth, expected_tools, is_refusal_case), chat_result in zip(prepared_cases, chat_results, strict=True):
        answer = str(chat_result["answer"])
        sources = list(chat_result["sources"] or [])
        tool_events = list(chat_result["tool_events"] or [])
        actual_tools = _selected_tools(tool_events)
        retrieved_sources = _extract_retrieved_sources(sources)
        cited_sources = _extract_cited_sources(answer)

        if is_refusal_case:
            citation_correctness = None
        elif cited_sources:
            matched = sum(1 for s in cited_sources if s in retrieved_sources)
            citation_correctness = matched / len(cited_sources)
        else:
            citation_correctness = 0.0 if sources else None

        retrieval_miss = len(sources) == 0
        strict_reject_ok = _is_strict_reject_answer(answer) if is_refusal_case or retrieval_miss else None
        tool_score = _tool_routing_score(expected_tools, actual_tools)

        if not is_refusal_case:
            data["user_input"].append(question)
            data["response"].append(answer)
            data["retrieved_contexts"].append([_source_content(source) for source in sources])
            data["reference"].append(ground_truth)

        per_case_stats.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "is_refusal_case": is_refusal_case,
                "retrieval_miss": retrieval_miss,
                "retrieved_sources": sorted(retrieved_sources),
                "cited_sources": cited_sources,
                "citation_correctness": citation_correctness,
                "strict_reject_ok": strict_reject_ok,
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "tool_routing_score": tool_score,
                "model": chat_result["model"],
            }
        )
        print(f"✅ 已处理: {question[:60]}...")

    if data["user_input"]:
        dataset = Dataset.from_dict(data)
    else:
        dataset = None

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    results = {}

    for metric in metrics:
        metric_name = getattr(metric, "name", metric.__class__.__name__)
        if dataset is None:
            print(f"⚠️ {metric_name}: 无可回答样本，跳过 RAGAS 计算")
            results[metric_name] = None
            continue
        try:
            print(f"正在计算 {metric_name} ...")
            result = evaluate(
                dataset=dataset,
                metrics=[metric],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                run_config=run_config,
            )
            raw_scores = result[metric_name]
            numeric_scores = pd.to_numeric(pd.Series(raw_scores), errors="coerce").dropna()
            score = float(numeric_scores.mean()) if not numeric_scores.empty else None
            results[metric_name] = score
            valid_count = int(numeric_scores.shape[0])
            total_count = len(raw_scores)
            if score is None:
                print(f"⚠️ {metric_name}: 无有效分数（有效样本 {valid_count}/{total_count}）")
            else:
                print(f"✅ {metric_name}: {score:.4f}（有效样本 {valid_count}/{total_count}）")
        except Exception as e:
            print(f"⚠️ {metric_name} 计算失败: {e}")
            results[metric_name] = None

    citation_correctness = _mean_or_none([x["citation_correctness"] for x in per_case_stats])
    results["citation_correctness"] = citation_correctness

    refusal_cases = [x for x in per_case_stats if x["is_refusal_case"]]
    if refusal_cases:
        results["refusal_correctness"] = _mean_or_none(
            [1.0 if x["strict_reject_ok"] else 0.0 for x in refusal_cases]
        )
    else:
        miss_cases = [x for x in per_case_stats if x["retrieval_miss"]]
        if miss_cases:
            results["refusal_correctness"] = _mean_or_none(
                [1.0 if x["strict_reject_ok"] else 0.0 for x in miss_cases]
            )
        else:
            results["refusal_correctness"] = None

    tool_scores = [x["tool_routing_score"] for x in per_case_stats]
    results["tool_routing_accuracy"] = _mean_or_none(tool_scores)

    print("\n📌 自定义指标：")
    for name in ("citation_correctness", "refusal_correctness", "tool_routing_accuracy"):
        value = results[name]
        print(f"   {name}: {value:.4f}" if value is not None else f"   {name}: None")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORT_DIR / "summary.csv"
    details_path = REPORT_DIR / "details.csv"
    legacy_summary_path = REPORT_DIR / "ragas_evaluation_report.csv"
    legacy_details_path = REPORT_DIR / "ragas_evaluation_details.csv"

    summary_df = pd.DataFrame([results])
    details_df = pd.DataFrame(per_case_stats)
    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)
    summary_df.to_csv(legacy_summary_path, index=False)
    details_df.to_csv(legacy_details_path, index=False)

    print("\n🎉 评估完成！最终平均指标：")
    for k, v in results.items():
        print(f"   {k}: {v}")
    print(f"📊 summary 已保存 → {summary_path}")
    print(f"📋 details 已保存 → {details_path}")
    return results


if __name__ == "__main__":
    run_ragas_evaluation()
