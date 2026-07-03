"""文档管理 API：上传与列表。"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.rag.retriever import ICRAGRetriever
from app.etl import ETLPipeline
from app.etl.chunker import ChunkStrategy
from app.infrastructure.database.models import Document, DocumentChunk
from app.infrastructure.database.session import get_async_session
from app.models.schemas import DocumentInfo, DocumentUploadResponse

router = APIRouter(tags=["documents"])


def _sync_pdf_to_data_dir(uploaded_path: Path, filename: str, doc_id: str) -> Path:
    settings = get_settings()
    data_dir = Path(settings.data_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    target = data_dir / filename
    if target.exists():
        target = data_dir / f"{Path(filename).stem}_{doc_id[:8]}{Path(filename).suffix}"

    shutil.copy2(uploaded_path, target)
    return target


def _document_text(document: Any) -> str:
    text = (getattr(document, "text", None) or "").strip()
    if text:
        return text

    getter = getattr(document, "get_content", None)
    if callable(getter):
        try:
            return (getter() or "").strip()
        except Exception:
            return ""
    return ""


def _document_metadata(document: Any) -> dict[str, Any]:
    metadata = getattr(document, "metadata", None)
    return dict(metadata or {}) if isinstance(metadata, dict) else {}


def _chunk_rows_from_vector_documents(documents: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, document in enumerate(documents):
        metadata = _document_metadata(document)
        chunk_id = str(
            metadata.get("chunk_id")
            or getattr(document, "node_id", "")
            or getattr(document, "id_", "")
        )
        rows.append(
            {
                "chunk_index": int(metadata.get("chunk_index") or i + 1),
                "content": _document_text(document),
                "vector_id": chunk_id or None,
                "meta": metadata,
            }
        )
    return rows


def _index_pdf_in_chroma(pdf_path: Path) -> tuple[str, list[Any]]:
    settings = get_settings()
    retriever = ICRAGRetriever(
        data_dir=settings.data_path,
        chroma_path=settings.chroma_path,
        collection_name=settings.chroma_collection_name,
        embedding_model=settings.embedding_model_path,
        embedding_device=settings.embedding_device,
        mismatch_strategy=settings.source_mismatch_strategy,
    )
    result = retriever.index_pdf(pdf_path)
    report = retriever.source_consistency_report
    if report is None:
        return "indexed", result.documents
    return report.reason, result.documents


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="上传的文件"),
    session: AsyncSession = Depends(get_async_session),
) -> DocumentUploadResponse:
    """上传文档并执行 ETL 分块后写入数据库。"""
    upload_root = Path("uploads")
    upload_root.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    safe_name = file.filename or "unnamed"
    dest = upload_root / f"{doc_id}_{safe_name}"

    try:
        raw = await file.read()
        await asyncio.to_thread(dest.write_bytes, raw)
    except Exception as exc:
        logger.exception("保存上传文件失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"保存文件失败: {exc!s}") from exc

    vector_status = "skipped"
    vector_message = "非 PDF 文件，未写入向量库"
    data_synced_path = ""
    chunk_rows: list[dict[str, Any]]
    is_pdf = safe_name.lower().endswith(".pdf") or file.content_type == "application/pdf"
    if is_pdf:
        try:
            synced = await asyncio.to_thread(_sync_pdf_to_data_dir, dest, safe_name, doc_id)
            data_synced_path = str(synced)
            vector_message, vector_documents = await asyncio.to_thread(
                _index_pdf_in_chroma,
                synced,
            )
            chunk_rows = _chunk_rows_from_vector_documents(vector_documents)
            vector_status = "indexed"
        except Exception as exc:
            logger.exception("向量入库失败: {}", exc)
            raise HTTPException(status_code=500, detail=f"向量入库失败: {exc!s}") from exc
    else:
        pipeline = ETLPipeline()
        try:
            etl = await pipeline.run_bytes(
                raw,
                filename=safe_name,
                mime_type=file.content_type,
                strategy=ChunkStrategy.IC_CUSTOM,
            )
        except Exception as exc:
            logger.exception("ETL 失败: {}", exc)
            raise HTTPException(status_code=422, detail=f"文档解析失败: {exc!s}") from exc
        chunk_rows = [
            {
                "chunk_index": i,
                "content": chunk_text,
                "vector_id": None,
                "meta": None,
            }
            for i, chunk_text in enumerate(etl.chunks)
        ]

    doc = Document(
        id=doc_id,
        filename=safe_name,
        mime_type=file.content_type,
        storage_path=str(dest),
        status="ready",
        meta={
            "chunk_count": len(chunk_rows),
            "vector_status": vector_status,
            "vector_message": vector_message,
            "data_synced_path": data_synced_path,
            "chunk_source": "chroma_index" if is_pdf else "etl_pipeline",
        },
    )
    session.add(doc)

    for row in chunk_rows:
        chunk = DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_index=row["chunk_index"],
            content=str(row["content"])[:65000],
            vector_id=row["vector_id"],
            meta=row["meta"],
        )
        session.add(chunk)

    await session.commit()

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=safe_name,
        status="ready",
        chunk_count=len(chunk_rows),
        message="上传并分块成功",
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(
    session: AsyncSession = Depends(get_async_session),
) -> list[DocumentInfo]:
    """列出已入库文档元数据。"""
    try:
        result = await session.execute(select(Document).order_by(Document.created_at.desc()))
        rows = result.scalars().all()
        out: list[DocumentInfo] = []
        for d in rows:
            out.append(
                DocumentInfo(
                    id=d.id,
                    filename=d.filename,
                    mime_type=d.mime_type,
                    status=d.status,
                    created_at=d.created_at.isoformat() if d.created_at else None,
                )
            )
        return out
    except Exception as exc:
        logger.exception("查询文档列表失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"查询失败: {exc!s}") from exc
