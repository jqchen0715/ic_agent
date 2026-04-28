# -*- coding: utf-8 -*-
from app.etl.chunker import ChunkStrategy, DocumentChunker


def test_fixed_chunks_respect_chunk_size() -> None:
    chunker = DocumentChunker(chunk_size=40, chunk_overlap=8)
    text = "alpha beta gamma delta " * 40

    chunks = chunker.chunk(text, strategy=ChunkStrategy.FIXED)

    assert chunks
    assert all(chunker._len(c) <= chunker.chunk_size for c in chunks)


def test_ic_custom_splits_oversized_verilog_block() -> None:
    chunker = DocumentChunker(chunk_size=60, chunk_overlap=10)
    verilog = "module demo;\n" + ("assign a = b & c;\n" * 80) + "endmodule\n"

    chunks = chunker.chunk(verilog, strategy=ChunkStrategy.IC_CUSTOM)

    assert len(chunks) > 1
    assert all(chunker._len(c) <= chunker.chunk_size for c in chunks)


def test_recursive_adds_overlap_prefix() -> None:
    chunker = DocumentChunker(chunk_size=30, chunk_overlap=6)
    text = (
        "first sentence for timing optimization.\n\n"
        "second sentence about setup hold closure.\n\n"
        "third sentence for retiming."
    )

    chunks = chunker.chunk(text, strategy=ChunkStrategy.RECURSIVE)

    assert len(chunks) >= 2
    tail = chunks[0][-6:]
    assert chunks[1].startswith(tail)
