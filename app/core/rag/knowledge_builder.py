"""统一知识库构建：PDF -> IC 分块 -> LlamaIndex Document -> Chroma。"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class KnowledgeBuildResult:
    """知识库构建结果。"""

    index: Any
    document_count: int
    pdf_count: int
    collection_name: str
    chroma_path: Path
    documents: list[Any] = field(default_factory=list)


def normalize_page_number(raw: Any) -> int | None:
    """把 0-based/1-based 页码元数据统一成 1-based 页码。"""
    if isinstance(raw, int):
        if raw < 0:
            return None
        return raw + 1 if raw == 0 else raw

    if isinstance(raw, str):
        text = raw.strip()
        if text.isdigit():
            page = int(text)
            return page + 1 if page == 0 else page

    return None


def first_present(*values: Any) -> Any:
    """返回第一个不是 None/空字符串的元数据值，保留 0 这类有效值。"""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def make_chunk_id(
    *,
    source: str,
    file_hash: str,
    page_num: int | None,
    chunk_index: int,
) -> str:
    """生成稳定且可放入 document_chunks.vector_id 的 chunk id。"""
    source_path = Path(source)
    suffix = source_path.suffix[:8]
    source_label = (source_path.stem or "pdf")[:56]
    page_label = str(page_num) if page_num is not None else "unknown"
    fingerprint = hashlib.sha1(
        f"{source}|{file_hash}|{page_label}|{chunk_index}".encode()
    ).hexdigest()[:10]
    return f"{source_label}{suffix}#p{page_label}#c{chunk_index}#{fingerprint}"


class KnowledgeBuilder:
    """唯一的 PDF 知识库构建器，供脚本、上传 API 和 Retriever 复用。"""

    def __init__(
        self,
        *,
        data_dir: str | Path,
        chroma_path: str | Path,
        collection_name: str,
        embed_model: Any,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.chroma_path = Path(chroma_path).resolve()
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        from app.etl.ic_text_splitter import ICCustomTextSplitter

        self._splitter = ICCustomTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def pdf_files(self, pdf_paths: Sequence[str | Path] | None = None) -> list[Path]:
        """返回参与建库的 PDF 文件列表。"""
        if pdf_paths is not None:
            files = [Path(path).resolve() for path in pdf_paths]
            missing = [str(path) for path in files if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"PDF 文件不存在: {missing}")
            pdf_files = sorted(path for path in files if path.suffix.lower() == ".pdf")
            if not pdf_files:
                raise FileNotFoundError(f"未找到 PDF 文件: {files}")
            return pdf_files

        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        pdf_files = sorted(path for path in self.data_dir.glob("*.pdf") if path.is_file())
        if not pdf_files:
            raise FileNotFoundError(f"数据目录下未找到 PDF: {self.data_dir}")
        return pdf_files

    def load_documents(self, pdf_paths: Sequence[str | Path] | None = None) -> list[Any]:
        """加载 PDF 并生成带统一 metadata 的 LlamaIndex Document。"""
        from langchain_community.document_loaders import PyPDFLoader
        from llama_index.core import Document

        pdf_files = self.pdf_files(pdf_paths)
        file_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            for path in pdf_files
        }

        lc_docs = []
        for pdf_file in pdf_files:
            lc_docs.extend(PyPDFLoader(str(pdf_file)).load())
        split_docs = self._splitter.split_documents(lc_docs)

        documents: list[Any] = []
        chunk_counts_by_source: dict[str, int] = {}
        for doc in split_docs:
            text = (doc.page_content or "").strip()
            if not text:
                continue

            metadata = dict(doc.metadata or {})
            source_path = Path(
                str(metadata.get("source") or metadata.get("file_path") or "unknown")
            )
            source = source_path.name
            page_num = normalize_page_number(
                first_present(
                    metadata.get("page"),
                    metadata.get("page_number"),
                    metadata.get("page_num"),
                    metadata.get("page_index"),
                )
            )

            chunk_counts_by_source[source] = chunk_counts_by_source.get(source, 0) + 1
            chunk_index = chunk_counts_by_source[source]
            file_hash = file_hashes.get(source, "")
            chunk_id = make_chunk_id(
                source=source,
                file_hash=file_hash,
                page_num=page_num,
                chunk_index=chunk_index,
            )
            metadata.update(
                {
                    "source": source,
                    "file_name": source,
                    "file_path": str(source_path),
                    "file_hash": file_hash,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "chunk_strategy": "ic_custom",
                }
            )
            if page_num is not None:
                metadata["page"] = page_num
                metadata["page_start"] = page_num
                metadata["page_end"] = page_num

            documents.append(Document(text=text, metadata=metadata, id_=chunk_id))

        if not documents:
            raise RuntimeError("未生成任何知识库切片")

        return documents

    @staticmethod
    def _matching_chunk_ids(collection: Any, field: str, value: str) -> set[str]:
        """返回符合单个 metadata 条件的 chunk ids。"""
        if not value:
            return set()

        try:
            data = collection.get(where={field: value}, include=["metadatas"])
        except Exception:
            return set()

        return {str(item) for item in data.get("ids", []) if item}

    def _delete_pdf_chunks(self, collection: Any, pdf_path: Path) -> int:
        """删除某个 PDF 在 collection 中已有的 chunks。"""
        source = pdf_path.name
        candidates = {
            ("source", source),
            ("source", str(pdf_path)),
            ("source", str(pdf_path.resolve())),
            ("file_name", source),
            ("file_path", str(pdf_path)),
            ("file_path", str(pdf_path.resolve())),
        }
        ids: set[str] = set()
        for metadata_field, value in candidates:
            ids.update(self._matching_chunk_ids(collection, metadata_field, value))
        if not ids:
            return 0

        collection.delete(ids=sorted(ids))
        return len(ids)

    def build_index(
        self,
        chroma_client: Any,
        *,
        recreate_collection: bool,
        show_progress: bool = True,
    ) -> KnowledgeBuildResult:
        """构建或重建 Chroma collection，并返回 VectorStoreIndex。"""
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore

        if recreate_collection:
            try:
                chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass

        collection = chroma_client.get_or_create_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        documents = self.load_documents()
        pdf_count = len({str(doc.metadata.get("source", "")) for doc in documents})

        logger.info(
            f"开始构建知识库: pdfs={pdf_count} "
            f"chunks={len(documents)} collection={self.collection_name}"
        )
        index = VectorStoreIndex(
            nodes=documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=show_progress,
        )
        return KnowledgeBuildResult(
            index=index,
            document_count=len(documents),
            pdf_count=pdf_count,
            collection_name=self.collection_name,
            chroma_path=self.chroma_path,
            documents=documents,
        )

    def index_pdf(
        self,
        chroma_client: Any,
        pdf_path: str | Path,
        *,
        show_progress: bool = True,
    ) -> KnowledgeBuildResult:
        """增量更新单个 PDF：删除该文件旧 chunks，然后写入新 chunks。"""
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore

        pdf = Path(pdf_path).resolve()
        documents = self.load_documents([pdf])
        collection = chroma_client.get_or_create_collection(self.collection_name)
        deleted = self._delete_pdf_chunks(collection, pdf)

        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        logger.info(
            f"开始增量更新知识库: pdf={pdf.name} deleted_chunks={deleted} "
            f"new_chunks={len(documents)} collection={self.collection_name}"
        )
        index = VectorStoreIndex(
            nodes=documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=show_progress,
        )
        return KnowledgeBuildResult(
            index=index,
            document_count=len(documents),
            pdf_count=1,
            collection_name=self.collection_name,
            chroma_path=self.chroma_path,
            documents=documents,
        )
