# -*- coding: utf-8 -*-
"""Milvus 长期记忆后端构造工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from app.core.memory.long_term import LongTermMemory

try:
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
except ImportError:  # pragma: no cover
    Collection = None  # type: ignore[assignment]
    CollectionSchema = None  # type: ignore[assignment]
    DataType = None  # type: ignore[assignment]
    FieldSchema = None  # type: ignore[assignment]
    connections = None  # type: ignore[assignment]
    utility = None  # type: ignore[assignment]


class SentenceTransformerMemoryEmbedder:
    """适配 LongTermMemory 的 sentence-transformers 嵌入器。"""

    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        model_path = Path(model_name)
        self._model_name = str(model_path.resolve()) if model_path.exists() else model_name
        self._device = device
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {}
        if self._device:
            kwargs["device"] = self._device
        self._model = SentenceTransformer(self._model_name, **kwargs)
        return self._model

    def embed_query(self, text: str) -> list[float]:
        model = self._load_model()
        vec = model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )[0]
        return [float(x) for x in vec.tolist()]

    def dimension(self) -> int:
        return len(self.embed_query("memory dimension probe"))


def build_milvus_long_term_memory(
    *,
    host: str,
    port: int | str,
    collection_name: str,
    embedding_model: str,
    embedding_device: str | None = None,
    user: str = "",
    password: str = "",
    alias: str = "memory",
) -> LongTermMemory:
    """连接 Milvus，确保记忆集合存在，并返回 LongTermMemory。"""
    if Collection is None or connections is None or utility is None:
        raise RuntimeError("未安装 pymilvus，无法启用 Milvus 记忆")

    conn_kwargs: dict[str, Any] = {
        "alias": alias,
        "host": host,
        "port": str(port),
    }
    if user:
        conn_kwargs["user"] = user
    if password:
        conn_kwargs["password"] = password

    connections.connect(**conn_kwargs)
    logger.info("已连接 Milvus 记忆库 {}:{} collection={}", host, port, collection_name)

    embedder = SentenceTransformerMemoryEmbedder(
        embedding_model,
        device=embedding_device,
    )
    if not utility.has_collection(collection_name, using=alias):
        dim = embedder.dimension()
        _create_memory_collection(collection_name, dim, alias)

    collection = Collection(collection_name, using=alias)
    _ensure_memory_index(collection)
    try:
        collection.load()
    except Exception as exc:  # pragma: no cover - Milvus 状态相关
        logger.warning("Milvus 记忆集合 load 失败，将在检索时重试: {}", exc)

    return LongTermMemory(collection, embedder)


def _create_memory_collection(collection_name: str, dim: int, alias: str) -> None:
    fields = [
        FieldSchema(
            name=LongTermMemory.pk_field,
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        ),
        FieldSchema(
            name=LongTermMemory.vector_field,
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
        ),
        FieldSchema(
            name=LongTermMemory.content_field,
            dtype=DataType.VARCHAR,
            max_length=65535,
        ),
        FieldSchema(
            name=LongTermMemory.session_field,
            dtype=DataType.VARCHAR,
            max_length=256,
        ),
        FieldSchema(
            name=LongTermMemory.meta_field,
            dtype=DataType.VARCHAR,
            max_length=65535,
        ),
    ]
    schema = CollectionSchema(
        fields=fields,
        description="Agent conversation long-term memory",
    )
    Collection(collection_name, schema=schema, using=alias)
    logger.info("已创建 Milvus 记忆集合 {} dim={}", collection_name, dim)


def _ensure_memory_index(collection: Any) -> None:
    try:
        if collection.indexes:
            return
    except Exception:
        pass

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    collection.create_index(
        field_name=LongTermMemory.vector_field,
        index_params=index_params,
    )
    logger.info("已创建 Milvus 记忆向量索引 collection={}", collection.name)
