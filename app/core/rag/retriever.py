# -*- coding: utf-8 -*-
"""IC 检索：LlamaIndex + Chroma，含 source 一致性检查与结构化输出。"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

from app.core.rag.reranker import Reranker
from app.etl.ic_text_splitter import ICCustomTextSplitter
from app.models.schemas import ICRetrievalResult


@dataclass
class SourceConsistencyReport:
    """data 目录 PDF 与 Chroma collection source 元数据的一致性报告。"""

    consistent: bool
    reason: str
    expected_sources: set[str]
    actual_sources: set[str]


def _normalize_source_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return Path(name).name.strip().lower()


def _normalize_page_number(metadata: dict[str, Any]) -> int | None:
    raw = (
        metadata.get("page")
        or metadata.get("page_number")
        or metadata.get("page_num")
        or metadata.get("page_index")
    )

    if isinstance(raw, int):
        if raw < 0:
            return None
        return raw + 1 if raw == 0 else raw

    if isinstance(raw, str):
        text = raw.strip()
        if text.isdigit():
            num = int(text)
            return num + 1 if num == 0 else num

    return None


def _page_label(metadata: dict[str, Any]) -> str:
    page_num = _normalize_page_number(metadata)
    if page_num is None:
        return "页码未知"
    return f"第{page_num}页"


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
        rerank_top_k: int = 10,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_device: str | None = None,
    ) -> None:
        base_dir = Path.cwd()

        self._data_dir = (Path(data_dir) if data_dir else base_dir / os.getenv("DATA_PATH", "data")).resolve()
        self._chroma_path = (
            Path(chroma_path) if chroma_path else base_dir / os.getenv("CHROMA_PATH", "chroma_db")
        ).resolve()

        self._collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "ic_expert")
        self._mismatch_strategy = (mismatch_strategy or os.getenv("SOURCE_MISMATCH_STRATEGY", "rebuild")).lower()
        if self._mismatch_strategy not in {"warn", "rebuild"}:
            self._mismatch_strategy = "rebuild"

        model_env = embedding_model or os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-m3")
        model_path = Path(model_env)
        model_name = str(model_path.resolve()) if model_path.exists() else str(model_env)

        self._embedding_device = embedding_device or os.getenv("EMBEDDING_DEVICE", "cpu")
        self._embed_model = HuggingFaceEmbedding(model_name=model_name, device=self._embedding_device)

        self._splitter = ICCustomTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._enable_reranker = enable_reranker
        self._retrieval_candidate_k = max(1, retrieval_candidate_k)
        self._rerank_top_k = max(1, rerank_top_k)
        self._reranker = Reranker(model_name=reranker_model, device=reranker_device) if enable_reranker else None
        self._index: VectorStoreIndex | None = None
        self._consistency_report: SourceConsistencyReport | None = None

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

    def _load_pdf_documents(self) -> list[Document]:
        if not self._data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self._data_dir}")

        pdf_files = sorted(self._data_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"数据目录下未找到 PDF: {self._data_dir}")

        loader = DirectoryLoader(str(self._data_dir), glob="*.pdf", loader_cls=PyPDFLoader)
        lc_docs = loader.load()
        split_docs = self._splitter.split_documents(lc_docs)

        docs: list[Document] = []
        for idx, doc in enumerate(split_docs, 1):
            metadata = dict(doc.metadata or {})
            source = Path(str(metadata.get("source") or metadata.get("file_path") or "unknown")).name
            metadata["source"] = source

            page_num = _normalize_page_number(metadata)
            if page_num is not None:
                metadata["page"] = page_num

            chunk_id = f"{source}#p{page_num if page_num is not None else 'unknown'}#c{idx}"
            metadata["chunk_id"] = chunk_id

            docs.append(Document(text=doc.page_content, metadata=metadata))

        return docs

    def _load_index_from_collection(self, collection: Any) -> VectorStoreIndex:
        vector_store = ChromaVectorStore(chroma_collection=collection)
        return VectorStoreIndex.from_vector_store(vector_store, embed_model=self._embed_model)

    def _build_index(self, chroma_client: Any, recreate_collection: bool) -> VectorStoreIndex:
        if recreate_collection:
            try:
                chroma_client.delete_collection(self._collection_name)
            except Exception:
                pass

        collection = chroma_client.get_or_create_collection(self._collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        docs = self._load_pdf_documents()
        logger.info(f"开始重建索引: docs={len(docs)} collection={self._collection_name}")

        return VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context,
            embed_model=self._embed_model,
            show_progress=True,
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
        """强制重建 Chroma 索引（用于上传新文档后立即可检索）。"""
        self._chroma_path.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(self._chroma_path))
        self._index = self._build_index(chroma_client, recreate_collection=True)
        try:
            collection = chroma_client.get_collection(self._collection_name)
            self._consistency_report = self._check_source_consistency(collection)
        except Exception:
            self._consistency_report = None
        return self._consistency_report

    def _run_reranker(self, query: str, results: list[ICRetrievalResult], top_k: int) -> list[ICRetrievalResult]:
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
            logger.warning(f"CrossEncoder 重排序失败，回退 dense 排序: {exc}")
            return results[:top_k]

    def retrieve(self, query: str, top_k: int = 3) -> list[ICRetrievalResult]:
        """执行 IC 检索并返回统一结构化结果。"""
        index = self._ensure_index()
        candidate_k = max(1, top_k, self._retrieval_candidate_k)
        retriever = index.as_retriever(similarity_top_k=candidate_k)
        nodes = retriever.retrieve(query)

        results: list[ICRetrievalResult] = []
        for idx, item in enumerate(nodes, 1):
            text = _extract_item_text(item).strip()
            if not text:
                continue

            node = getattr(item, "node", None)
            metadata = dict(getattr(node, "metadata", {}) or {})

            source = Path(str(metadata.get("source") or metadata.get("file_name") or "unknown")).name
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
                    score=score,
                    chunk_id=chunk_id,
                )
            )

        if not results:
            return []

        rerank_top_k = min(max(top_k, self._rerank_top_k), len(results))
        reranked = self._run_reranker(query, results, rerank_top_k) if self._enable_reranker else results
        return reranked[:top_k]


# 兼容旧命名，避免已有调用方 import MultiRetriever 失效。
MultiRetriever = ICRAGRetriever
