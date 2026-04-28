# -*- coding: utf-8 -*-
"""构建 IC 知识库：PDF -> IC 定制分块 -> Chroma 持久化。"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.etl.chunker import ChunkStrategy, DocumentChunker
from app.etl.ic_text_splitter import ICCustomTextSplitter
from app.etl.parser import DocumentParser


def _load_runtime_defaults() -> dict[str, str]:
    """优先读取 app.config(.env)，失败时回退环境变量/硬编码默认值。"""
    defaults = {
        "data_dir": str(PROJECT_ROOT / "data"),
        "chroma_path": str(PROJECT_ROOT / "chroma_db"),
        "collection_name": "ic_expert",
        "embedding_model": "BAAI/bge-m3",
        "embedding_device": "cpu",
    }
    try:
        from app.config import get_settings

        s = get_settings()
        defaults["data_dir"] = str(Path(s.data_path))
        defaults["chroma_path"] = str(Path(s.chroma_path))
        defaults["collection_name"] = s.chroma_collection_name
        defaults["embedding_model"] = s.embedding_model_path
        defaults["embedding_device"] = s.embedding_device
        return defaults
    except Exception:
        pass

    defaults["data_dir"] = os.getenv("DATA_PATH", defaults["data_dir"])
    defaults["chroma_path"] = os.getenv("CHROMA_PATH", defaults["chroma_path"])
    defaults["collection_name"] = os.getenv("CHROMA_COLLECTION_NAME", defaults["collection_name"])
    defaults["embedding_model"] = os.getenv("EMBEDDING_MODEL_PATH", defaults["embedding_model"])
    defaults["embedding_device"] = os.getenv("EMBEDDING_DEVICE", defaults["embedding_device"])
    return defaults


def parse_args() -> argparse.Namespace:
    defaults = _load_runtime_defaults()
    parser = argparse.ArgumentParser(description="构建 IC PDF 知识库到 Chroma。")
    parser.add_argument("--data-dir", type=Path, default=Path(defaults["data_dir"]), help="PDF 数据目录")
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path(defaults["chroma_path"]),
        help="Chroma 持久化目录",
    )
    parser.add_argument("--collection-name", default=defaults["collection_name"], help="Chroma collection 名称")
    parser.add_argument(
        "--embedding-model",
        default=defaults["embedding_model"],
        help="HuggingFace embedding 模型（默认优先取 .env 的 EMBEDDING_MODEL_PATH）",
    )
    parser.add_argument(
        "--embedding-device",
        default=defaults["embedding_device"],
        help="embedding 运行设备: cpu/mps/cuda",
    )
    parser.add_argument("--chunk-size", type=int, default=800, help="分块 token/字符上限")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="分块重叠 token/字符数")
    parser.add_argument("--reset", action="store_true", help="构建前删除已有 chroma_db 目录")
    return parser.parse_args()


def normalize_page_number(raw: Any) -> int | None:
    if isinstance(raw, int):
        return raw + 1 if raw == 0 else raw
    if isinstance(raw, str) and raw.strip().isdigit():
        page = int(raw.strip())
        return page + 1 if page == 0 else page
    return None


def load_pdf_text(path: Path, parser: DocumentParser) -> tuple[str, dict[str, str]]:
    parsed = parser.parse_file(path, mime_type="application/pdf")
    text = parsed.text.strip()
    if not text:
        raise ValueError(f"PDF 未解析出文本: {path}")
    return text, parsed.meta


def chunk_pdf(
    path: Path,
    parser: DocumentParser,
    chunker: DocumentChunker,
    ic_splitter: ICCustomTextSplitter,
) -> list[Document]:
    text, parsed_meta = load_pdf_text(path, parser)

    ic_chunks = ic_splitter.split_text(text)
    chunks = chunker.chunk("\n\n".join(ic_chunks), strategy=ChunkStrategy.IC_CUSTOM)

    documents: list[Document] = []
    source = path.name
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    page_num = normalize_page_number(parsed_meta.get("page"))

    for idx, chunk in enumerate(chunks, 1):
        metadata: dict[str, Any] = {
            "source": source,
            "file_name": source,
            "file_path": str(path),
            "file_hash": file_hash,
            "chunk_id": f"{source}#c{idx}",
            "chunk_index": idx,
            "chunk_strategy": ChunkStrategy.IC_CUSTOM.value,
        }
        if page_num is not None:
            metadata["page"] = page_num
        if parsed_meta.get("pages"):
            metadata["total_pages"] = parsed_meta["pages"]

        documents.append(Document(text=chunk, metadata=metadata))

    return documents


def build_knowledge_base(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    chroma_path = args.chroma_path.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"数据目录下未找到 PDF: {data_dir}")

    if args.reset and chroma_path.exists():
        shutil.rmtree(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)

    parser = DocumentParser()
    chunker = DocumentChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    ic_splitter = ICCustomTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    documents: list[Document] = []
    for pdf_path in pdf_files:
        pdf_documents = chunk_pdf(pdf_path, parser, chunker, ic_splitter)
        documents.extend(pdf_documents)
        print(f"已处理 {pdf_path.name}: chunks={len(pdf_documents)}")

    if not documents:
        raise RuntimeError("未生成任何知识库切片")

    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    if args.reset:
        collection = chroma_client.get_or_create_collection(args.collection_name)
    else:
        try:
            chroma_client.delete_collection(args.collection_name)
        except Exception:
            pass
        collection = chroma_client.get_or_create_collection(args.collection_name)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = HuggingFaceEmbedding(model_name=args.embedding_model, device=args.embedding_device)
    print(f"embedding_model={args.embedding_model}")
    print(f"embedding_device={args.embedding_device}")

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print(
        "知识库构建完成: "
        f"pdfs={len(pdf_files)} chunks={len(documents)} "
        f"collection={args.collection_name} chroma_path={chroma_path}"
    )


def main() -> None:
    args = parse_args()
    build_knowledge_base(args)


if __name__ == "__main__":
    main()
