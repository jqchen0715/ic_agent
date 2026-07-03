"""IC 检索：LlamaIndex + Chroma，含 source 一致性检查与结构化输出。"""

from __future__ import annotations

import asyncio
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

from app.core.rag.knowledge_builder import (
    KnowledgeBuilder,
    KnowledgeBuildResult,
    first_present,
    normalize_page_number,
)
from app.core.rag.reranker import Reranker
from app.models.schemas import ICRetrievalResult


@dataclass
class SourceConsistencyReport:
    """data 目录 PDF 与 Chroma collection source 元数据的一致性报告。"""

    consistent: bool
    reason: str
    expected_sources: set[str]
    actual_sources: set[str]


_KEYWORD_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+")


@dataclass
class KeywordCorpusItem:
    """Chroma 中可用于 BM25 的单个 chunk。"""

    id: str
    text: str
    metadata: dict[str, Any]


def _normalize_source_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return Path(name).name.strip().lower()


def _normalize_page_number(metadata: dict[str, Any]) -> int | None:
    return normalize_page_number(
        first_present(
            metadata.get("page"),
            metadata.get("page_number"),
            metadata.get("page_num"),
            metadata.get("page_index"),
        )
    )


def _page_label(metadata: dict[str, Any]) -> str:
    page_num = _normalize_page_number(metadata)
    if page_num is None:
        return "页码未知"
    return f"第{page_num}页"


def _keyword_tokens(text: str) -> list[str]:
    """面向 IC/代码文档的轻量分词，保留精确标识符和中文短语。"""
    tokens: list[str] = []
    for raw in _KEYWORD_TOKEN_RE.findall(text or ""):
        token = raw.lower()
        if not token:
            continue
        tokens.append(token)

        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 2:
            tokens.extend(token[i : i + 2] for i in range(len(token) - 1))
            if len(token) > 3:
                tokens.extend(token[i : i + 3] for i in range(len(token) - 2))
    return tokens


def _keyword_query_terms(query: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for token in _keyword_tokens(query):
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _keyword_result_key(item: ICRetrievalResult) -> tuple[str, str, str]:
    return (
        str(item.chunk_id or ""),
        _normalize_source_name(item.source),
        str(item.page or ""),
    )


def _extract_item_text(item: NodeWithScore) -> str:
    text = (getattr(item, "text", None) or "").strip()
    if text:
        return text

    node = getattr(item, "node", None)
    if node is None:
        return ""

    getter = getattr(node, "get_content", None)
    if callable(getter):
        try:
            text = (getter() or "").strip()
            if text:
                return text
        except Exception:
            pass

    return (getattr(node, "text", None) or "").strip()


class ICRAGRetriever:
    """IC 领域检索器：自动维护 Chroma 索引并返回标准结构化结果。"""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        chroma_path: str | Path | None = None,
        collection_name: str = "ic_expert",
        embedding_model: str | Path | None = None,
        embedding_device: str | None = None,
        mismatch_strategy: str = "rebuild",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        enable_reranker: bool = True,
        retrieval_candidate_k: int = 20,
        enable_keyword_retrieval: bool = True,
        keyword_candidate_k: int = 20,
        dense_weight: float = 0.65,
        keyword_weight: float = 0.55,
        rerank_top_k: int = 10,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_device: str | None = None,
    ) -> None:
        base_dir = Path.cwd()

        self._data_dir = (
            Path(data_dir) if data_dir else base_dir / os.getenv("DATA_PATH", "data")
        ).resolve()
        self._chroma_path = (
            Path(chroma_path) if chroma_path else base_dir / os.getenv("CHROMA_PATH", "chroma_db")
        ).resolve()

        self._collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "ic_expert")
        self._mismatch_strategy = (
            mismatch_strategy or os.getenv("SOURCE_MISMATCH_STRATEGY", "rebuild")
        ).lower()
        if self._mismatch_strategy not in {"warn", "rebuild"}:
            self._mismatch_strategy = "rebuild"

        model_env = embedding_model or os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-m3")
        model_path = Path(model_env)
        model_name = str(model_path.resolve()) if model_path.exists() else str(model_env)

        self._embedding_device = embedding_device or os.getenv("EMBEDDING_DEVICE", "cpu")
        self._embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            device=self._embedding_device,
        )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._enable_reranker = enable_reranker
        self._retrieval_candidate_k = max(1, retrieval_candidate_k)
        self._enable_keyword_retrieval = enable_keyword_retrieval
        self._keyword_candidate_k = max(1, keyword_candidate_k)
        self._dense_weight = max(0.0, dense_weight)
        self._keyword_weight = max(0.0, keyword_weight)
        self._rerank_top_k = max(1, rerank_top_k)
        self._reranker = (
            Reranker(model_name=reranker_model, device=reranker_device)
            if enable_reranker
            else None
        )
        self._index: VectorStoreIndex | None = None
        self._consistency_report: SourceConsistencyReport | None = None
        self._keyword_corpus: list[KeywordCorpusItem] | None = None

    @property
    def source_consistency_report(self) -> SourceConsistencyReport | None:
        return self._consistency_report

    def _expected_pdf_sources(self) -> set[str]:
        if not self._data_dir.exists():
            return set()
        return {
            _normalize_source_name(path.name)
            for path in self._data_dir.glob("*.pdf")
            if path.is_file()
        }

    def _collection_pdf_sources(self, chroma_collection: Any) -> set[str]:
        try:
            all_data = chroma_collection.get(include=["metadatas"])
        except Exception:
            return set()

        def _iter_metadata(items: Any):
            if isinstance(items, dict):
                yield items
                return
            if isinstance(items, list):
                for item in items:
                    yield from _iter_metadata(item)

        sources: set[str] = set()
        for md in _iter_metadata(all_data.get("metadatas", [])):
            source = _normalize_source_name(md.get("source") or md.get("file_name") or "")
            if source:
                sources.add(source)
        return sources

    def _check_source_consistency(self, chroma_collection: Any) -> SourceConsistencyReport:
        expected = self._expected_pdf_sources()
        actual = self._collection_pdf_sources(chroma_collection)

        if not expected:
            return SourceConsistencyReport(
                consistent=True,
                reason="data 目录为空，跳过一致性检查",
                expected_sources=expected,
                actual_sources=actual,
            )

        if not actual:
            return SourceConsistencyReport(
                consistent=False,
                reason="向量库为空或缺少 source 元数据",
                expected_sources=expected,
                actual_sources=actual,
            )

        if expected != actual:
            extra = sorted(actual - expected)
            missing = sorted(expected - actual)
            return SourceConsistencyReport(
                consistent=False,
                reason=f"source 不一致 (extra={extra}, missing={missing})",
                expected_sources=expected,
                actual_sources=actual,
            )

        return SourceConsistencyReport(
            consistent=True,
            reason="source 一致",
            expected_sources=expected,
            actual_sources=actual,
        )

    def _load_index_from_collection(self, collection: Any) -> VectorStoreIndex:
        vector_store = ChromaVectorStore(chroma_collection=collection)
        return VectorStoreIndex.from_vector_store(vector_store, embed_model=self._embed_model)

    def _build_index(self, chroma_client: Any, recreate_collection: bool) -> VectorStoreIndex:
        builder = KnowledgeBuilder(
            data_dir=self._data_dir,
            chroma_path=self._chroma_path,
            collection_name=self._collection_name,
            embed_model=self._embed_model,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        result = builder.build_index(
            chroma_client,
            recreate_collection=recreate_collection,
            show_progress=True,
        )
        self._keyword_corpus = None
        return result.index

    def _new_builder(self) -> KnowledgeBuilder:
        return KnowledgeBuilder(
            data_dir=self._data_dir,
            chroma_path=self._chroma_path,
            collection_name=self._collection_name,
            embed_model=self._embed_model,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    def _ensure_index(self) -> VectorStoreIndex:
        if self._index is not None:
            return self._index

        self._chroma_path.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(self._chroma_path))

        try:
            collection = chroma_client.get_collection(self._collection_name)
            report = self._check_source_consistency(collection)
            self._consistency_report = report

            if not report.consistent:
                message = f"检测到知识库 source 不一致: {report.reason}"
                if self._mismatch_strategy == "rebuild":
                    logger.warning(f"{message}，执行重建")
                    self._index = self._build_index(chroma_client, recreate_collection=True)

                    refreshed = chroma_client.get_collection(self._collection_name)
                    self._consistency_report = self._check_source_consistency(refreshed)
                else:
                    logger.warning(f"{message}，仅提示，不自动重建")
                    self._index = self._load_index_from_collection(collection)
            else:
                logger.info(f"使用已有 Chroma 索引: {report.reason}")
                self._index = self._load_index_from_collection(collection)

        except Exception as exc:
            logger.info(f"未发现可用索引，开始首次构建: {exc}")
            self._index = self._build_index(chroma_client, recreate_collection=False)
            try:
                collection = chroma_client.get_collection(self._collection_name)
                self._consistency_report = self._check_source_consistency(collection)
            except Exception:
                self._consistency_report = None

        return self._index

    def rebuild_index(self) -> SourceConsistencyReport | None:
        """强制重建 Chroma 索引（用于手动修复/全量刷新）。"""
        self._chroma_path.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(self._chroma_path))
        self._index = self._build_index(chroma_client, recreate_collection=True)
        self._keyword_corpus = None
        try:
            collection = chroma_client.get_collection(self._collection_name)
            self._consistency_report = self._check_source_consistency(collection)
        except Exception:
            self._consistency_report = None
        return self._consistency_report

    def index_pdf(self, pdf_path: str | Path) -> KnowledgeBuildResult:
        """增量更新单个 PDF 到 Chroma（用于上传后立即可检索）。"""
        self._chroma_path.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(self._chroma_path))
        result = self._new_builder().index_pdf(
            chroma_client,
            pdf_path,
            show_progress=True,
        )
        self._index = result.index
        self._keyword_corpus = None
        try:
            collection = chroma_client.get_collection(self._collection_name)
            self._consistency_report = self._check_source_consistency(collection)
        except Exception:
            self._consistency_report = None
        return result

    def _load_keyword_corpus(self) -> list[KeywordCorpusItem]:
        if self._keyword_corpus is not None:
            return self._keyword_corpus

        try:
            chroma_client = chromadb.PersistentClient(path=str(self._chroma_path))
            collection = chroma_client.get_collection(self._collection_name)
            data = collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            logger.warning(f"BM25 关键词召回读取 Chroma 失败: {exc}")
            self._keyword_corpus = []
            return self._keyword_corpus

        ids = data.get("ids", []) or []
        documents = data.get("documents", []) or []
        metadatas = data.get("metadatas", []) or []
        corpus: list[KeywordCorpusItem] = []
        for idx, text in enumerate(documents):
            content = str(text or "").strip()
            if not content:
                continue
            metadata = (
                metadatas[idx]
                if idx < len(metadatas) and isinstance(metadatas[idx], dict)
                else {}
            )
            item_id = str(ids[idx]) if idx < len(ids) else str(metadata.get("chunk_id") or idx)
            corpus.append(KeywordCorpusItem(id=item_id, text=content, metadata=dict(metadata)))

        self._keyword_corpus = corpus
        return corpus

    def _keyword_retrieve(self, query: str, candidate_k: int) -> list[ICRetrievalResult]:
        terms = _keyword_query_terms(query)
        if not terms:
            return []

        corpus = self._load_keyword_corpus()
        if not corpus:
            return []

        tokenized = [_keyword_tokens(item.text) for item in corpus]
        lengths = [len(tokens) for tokens in tokenized]
        avgdl = sum(lengths) / len(lengths) if lengths else 1.0
        avgdl = max(avgdl, 1.0)

        doc_freq: Counter[str] = Counter()
        for tokens in tokenized:
            doc_freq.update(set(tokens))

        term_counts = Counter(terms)
        k1 = 1.5
        b = 0.75
        total_docs = len(corpus)
        scored: list[tuple[float, KeywordCorpusItem]] = []
        for item, tokens, doc_len in zip(corpus, tokenized, lengths, strict=True):
            if not tokens:
                continue

            counts = Counter(tokens)
            score = 0.0
            for term, query_tf in term_counts.items():
                tf = counts.get(term, 0)
                if tf <= 0:
                    continue
                df = max(1, doc_freq.get(term, 0))
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1.0 - b + b * doc_len / avgdl)
                score += query_tf * idf * (tf * (k1 + 1.0)) / denom

            if score <= 0:
                continue

            lower_text = item.text.lower()
            exact_bonus = sum(0.2 for term in term_counts if len(term) > 1 and term in lower_text)
            scored.append((score + exact_bonus, item))

        if not scored:
            return []

        scored.sort(key=lambda pair: pair[0], reverse=True)
        max_score = max(score for score, _ in scored) or 1.0
        results: list[ICRetrievalResult] = []
        for raw_score, item in scored[:candidate_k]:
            metadata = item.metadata
            source = Path(
                str(metadata.get("source") or metadata.get("file_name") or "unknown")
            ).name
            chunk_id = str(metadata.get("chunk_id") or item.id)
            results.append(
                ICRetrievalResult(
                    content=item.text,
                    source=source,
                    page=_page_label(metadata),
                    score=(raw_score / max_score) * self._keyword_weight,
                    chunk_id=chunk_id,
                )
            )
        return results

    def _dense_retrieve(self, query: str, candidate_k: int) -> list[ICRetrievalResult]:
        index = self._ensure_index()
        retriever = index.as_retriever(similarity_top_k=candidate_k)
        nodes = retriever.retrieve(query)

        results: list[ICRetrievalResult] = []
        for idx, item in enumerate(nodes, 1):
            text = _extract_item_text(item).strip()
            if not text:
                continue

            node = getattr(item, "node", None)
            metadata = dict(getattr(node, "metadata", {}) or {})

            source = Path(
                str(metadata.get("source") or metadata.get("file_name") or "unknown")
            ).name
            page = _page_label(metadata)
            score = float(item.score or 0.0)
            chunk_id = str(
                metadata.get("chunk_id")
                or getattr(node, "node_id", "")
                or f"{source}#hit{idx}"
            )

            results.append(
                ICRetrievalResult(
                    content=text,
                    source=source,
                    page=page,
                    score=score * self._dense_weight,
                    chunk_id=chunk_id,
                )
            )
        return results

    def _merge_hybrid_results(
        self,
        dense_results: list[ICRetrievalResult],
        keyword_results: list[ICRetrievalResult],
    ) -> list[ICRetrievalResult]:
        merged: dict[tuple[str, str, str], ICRetrievalResult] = {}
        route_counts: dict[tuple[str, str, str], int] = {}

        for item in [*dense_results, *keyword_results]:
            key = _keyword_result_key(item)
            prev = merged.get(key)
            route_counts[key] = route_counts.get(key, 0) + 1
            if prev is None:
                merged[key] = item.model_copy(deep=True)
                continue

            combined_score = max(float(prev.score or 0.0), float(item.score or 0.0))
            if route_counts[key] > 1:
                combined_score += 0.12
            if len(item.content or "") > len(prev.content or ""):
                prev.content = item.content
            prev.score = combined_score

        return sorted(
            merged.values(),
            key=lambda item: float(item.score or 0.0),
            reverse=True,
        )

    def _run_reranker(
        self,
        query: str,
        results: list[ICRetrievalResult],
        top_k: int,
    ) -> list[ICRetrievalResult]:
        if self._reranker is None or len(results) <= 1:
            return results[:top_k]

        async def _rerank() -> list[ICRetrievalResult]:
            return await self._reranker.rerank(query, results, top_k=top_k)

        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(_rerank())

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: asyncio.run(_rerank()))
                return future.result()
        except Exception as exc:
            logger.warning(f"CrossEncoder 重排序失败，回退 hybrid 排序: {exc}")
            return results[:top_k]

    def retrieve(self, query: str, top_k: int = 3) -> list[ICRetrievalResult]:
        """执行 IC hybrid 检索并返回统一结构化结果。"""
        candidate_k = max(1, top_k, self._retrieval_candidate_k)
        dense_results = self._dense_retrieve(query, candidate_k)
        keyword_results = (
            self._keyword_retrieve(query, max(top_k, self._keyword_candidate_k))
            if self._enable_keyword_retrieval
            else []
        )
        results = self._merge_hybrid_results(dense_results, keyword_results)

        if not results:
            return []

        rerank_top_k = min(max(top_k, self._rerank_top_k), len(results))
        reranked = (
            self._run_reranker(query, results, rerank_top_k)
            if self._enable_reranker
            else results
        )
        return reranked[:top_k]


# 兼容旧命名，避免已有调用方 import MultiRetriever 失效。
MultiRetriever = ICRAGRetriever
