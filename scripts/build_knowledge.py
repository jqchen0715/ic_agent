"""构建 IC 知识库：PDF -> IC 定制分块 -> Chroma 持久化。"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.rag.knowledge_builder import KnowledgeBuilder  # noqa: E402


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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(defaults["data_dir"]),
        help="PDF 数据目录",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path(defaults["chroma_path"]),
        help="Chroma 持久化目录",
    )
    parser.add_argument(
        "--collection-name",
        default=defaults["collection_name"],
        help="Chroma collection 名称",
    )
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


def build_knowledge_base(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    chroma_path = args.chroma_path.resolve()

    if args.reset and chroma_path.exists():
        shutil.rmtree(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    embed_model = HuggingFaceEmbedding(
        model_name=args.embedding_model,
        device=args.embedding_device,
    )

    print(f"embedding_model={args.embedding_model}")
    print(f"embedding_device={args.embedding_device}")
    builder = KnowledgeBuilder(
        data_dir=data_dir,
        chroma_path=chroma_path,
        collection_name=args.collection_name,
        embed_model=embed_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    result = builder.build_index(
        chroma_client,
        recreate_collection=True,
        show_progress=True,
    )

    print(
        "知识库构建完成: "
        f"pdfs={result.pdf_count} chunks={result.document_count} "
        f"collection={args.collection_name} chroma_path={chroma_path}"
    )


def main() -> None:
    args = parse_args()
    build_knowledge_base(args)


if __name__ == "__main__":
    main()
