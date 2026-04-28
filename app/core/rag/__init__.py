# -*- coding: utf-8 -*-
"""RAG 子系统：检索、重排、生成。"""

from app.core.rag.citation_rewriter import CitationRewriteResult, rewrite_answer_citations
try:  # pragma: no cover - 可选依赖缺失时保持最小可用导入
    from app.core.rag.retriever import ICRAGRetriever, MultiRetriever, SourceConsistencyReport
except Exception:  # noqa: BLE001
    ICRAGRetriever = None  # type: ignore[assignment]
    MultiRetriever = None  # type: ignore[assignment]
    SourceConsistencyReport = None  # type: ignore[assignment]

try:  # pragma: no cover - 可选依赖缺失时保持最小可用导入
    from app.core.rag.generator import RAGGenerator
except Exception:  # noqa: BLE001
    RAGGenerator = None  # type: ignore[assignment]

try:  # pragma: no cover
    from app.core.rag.reranker import Reranker
except Exception:  # noqa: BLE001
    Reranker = None  # type: ignore[assignment]

__all__ = [
    "ICRAGRetriever",
    "MultiRetriever",
    "SourceConsistencyReport",
    "CitationRewriteResult",
    "rewrite_answer_citations",
    "Reranker",
    "RAGGenerator",
]
