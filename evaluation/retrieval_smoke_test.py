# -*- coding: utf-8 -*-
"""IC 检索冒烟测试：验证 Chroma + LlamaIndex 检索链路可用。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_runtime_defaults() -> dict[str, str]:
    """优先读取 app.config(.env)，失败时回退环境变量/硬编码默认值。"""
    defaults = {
        "data_dir": str(PROJECT_ROOT / "data"),
        "chroma_path": str(PROJECT_ROOT / "chroma_db"),
        "collection_name": "ic_expert",
        "embedding_model": "BAAI/bge-m3",
        "mismatch_strategy": "rebuild",
    }
    try:
        from app.config import get_settings

        s = get_settings()
        defaults["data_dir"] = str(Path(s.data_path))
        defaults["chroma_path"] = str(Path(s.chroma_path))
        defaults["collection_name"] = s.chroma_collection_name
        defaults["embedding_model"] = s.embedding_model_path
        defaults["mismatch_strategy"] = s.source_mismatch_strategy
        return defaults
    except Exception:
        pass

    defaults["data_dir"] = os.getenv("DATA_PATH", defaults["data_dir"])
    defaults["chroma_path"] = os.getenv("CHROMA_PATH", defaults["chroma_path"])
    defaults["collection_name"] = os.getenv("CHROMA_COLLECTION_NAME", defaults["collection_name"])
    defaults["embedding_model"] = os.getenv("EMBEDDING_MODEL_PATH", defaults["embedding_model"])
    defaults["mismatch_strategy"] = os.getenv("SOURCE_MISMATCH_STRATEGY", defaults["mismatch_strategy"])
    return defaults


def parse_args() -> argparse.Namespace:
    defaults = _load_runtime_defaults()
    parser = argparse.ArgumentParser(description="Run IC retrieval smoke test.")
    parser.add_argument(
        "--query",
        default="乘法器时序优化有哪些方法？",
        help="检索问题",
    )
    parser.add_argument(
        "--data-dir",
        default=defaults["data_dir"],
        help="PDF 知识库目录（默认: ./data）",
    )
    parser.add_argument(
        "--chroma-path",
        default=defaults["chroma_path"],
        help="Chroma 持久化目录（默认: ./chroma_db）",
    )
    parser.add_argument(
        "--collection-name",
        default=defaults["collection_name"],
        help="Chroma collection 名",
    )
    parser.add_argument(
        "--embedding-model",
        default=defaults["embedding_model"],
        help="Embedding 模型路径或模型名（默认优先取 .env 的 EMBEDDING_MODEL_PATH）",
    )
    parser.add_argument(
        "--mismatch-strategy",
        choices=["warn", "rebuild"],
        default=defaults["mismatch_strategy"] if defaults["mismatch_strategy"] in {"warn", "rebuild"} else "rebuild",
        help="source 不一致处理策略",
    )
    parser.add_argument("--top-k", type=int, default=3, help="返回结果条数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from app.core.rag.retriever import ICRAGRetriever
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "unknown")
        print(
            json.dumps(
                {
                    "error": f"缺少运行依赖: {missing}",
                    "hint": "请先执行: pip install -r requirements.txt",
                },
                ensure_ascii=False,
            )
        )
        return 2

    retriever = ICRAGRetriever(
        data_dir=args.data_dir,
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        mismatch_strategy=args.mismatch_strategy,
    )

    results = retriever.retrieve(args.query, top_k=max(1, args.top_k))
    if not results:
        print("[FAIL] 未检索到任何结果")
        return 1

    report = retriever.source_consistency_report
    if report is not None:
        print(
            json.dumps(
                {
                    "source_consistency": {
                        "consistent": report.consistent,
                        "reason": report.reason,
                        "expected_count": len(report.expected_sources),
                        "actual_count": len(report.actual_sources),
                    }
                },
                ensure_ascii=False,
            )
        )

    print(json.dumps({"query": args.query, "result_count": len(results)}, ensure_ascii=False))

    for item in results[: max(1, min(args.top_k, 3))]:
        row = {
            "content": item.content,
            "source": item.source,
            "page": item.page,
            "score": item.score,
            "chunk_id": item.chunk_id,
        }
        print(json.dumps(row, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
