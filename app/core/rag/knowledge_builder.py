"""统一知识库构建：PDF -> IC 分块 -> LlamaIndex Document -> Chroma。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
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

    def pdf_files(self) -> list[Path]:
        """返回参与建库的 PDF 文件列表。"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        pdf_files = sorted(path for path in self.data_dir.glob("*.pdf") if path.is_file())
        if not pdf_files:
            raise FileNotFoundError(f"数据目录下未找到 PDF: {self.data_dir}")
        return pdf_files

    def load_documents(self) -> list[Any]:
        """加载 PDF 并生成带统一 metadata 的 LlamaIndex Document。"""
        from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
        from llama_index.core import Document

        pdf_files = self.pdf_files()
        file_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            for path in pdf_files
        }

        loader = DirectoryLoader(str(self.data_dir), glob="*.pdf", loader_cls=PyPDFLoader)
        lc_docs = loader.load()
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
            metadata.update(
                {
                    "source": source,
                    "file_name": source,
                    "file_path": str(source_path),
                    "file_hash": file_hashes.get(source, ""),
                    "chunk_id": (
                        f"{source}#p{page_num if page_num is not None else 'unknown'}"
                        f"#c{chunk_index}"
                    ),
                    "chunk_index": chunk_index,
                    "chunk_strategy": "ic_custom",
                }
            )
            if page_num is not None:
                metadata["page"] = page_num
                metadata["page_start"] = page_num
                metadata["page_end"] = page_num

            documents.append(Document(text=text, metadata=metadata))

        if not documents:
            raise RuntimeError("未生成任何知识库切片")

        return documents

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
        index = VectorStoreIndex.from_documents(
            documents,
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
        )
