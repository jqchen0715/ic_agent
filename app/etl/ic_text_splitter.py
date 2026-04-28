# -*- coding: utf-8 -*-
"""IC 文档定制分块器：优先保留 Verilog 代码块、章节和时序图边界。"""

from __future__ import annotations

import re
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter


class ICCustomTextSplitter(TextSplitter):
    """针对 IC 资料的分块器，优先保持代码和结构化段落语义完整。"""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
        length_function: Callable[[str], int] | None = None,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._length_function = length_function or len
        self.separators = separators or [
            r"(?<=\bendmodule)",
            r"(?=\bmodule\s+)",
            r"```verilog[\s\S]*?```",
            r"(?<=\n)(图\d+[\u4e00-\u9fa5A-Za-z\s]*?(时序图|波形图|Timing Diagram))(?=\n)",
            r"(?<=\n)(\d+(\.\d+)+[\u4e00-\u9fa5\s]*?[:：])",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ]
        self.fine_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=self._length_function,
        )

    def split_text(self, text: str) -> list[str]:
        """先粗切分再细切分，避免长段落截断语义。"""
        raw = (text or "").strip()
        if not raw:
            return []

        coarse_blocks = self._split_verilog_blocks(raw)
        chunks: list[str] = []

        for block in coarse_blocks:
            if self._is_verilog_block(block):
                if self._length_function(block) > self._chunk_size:
                    chunks.extend(self.fine_splitter.split_text(block))
                else:
                    chunks.append(block)
                continue

            chapter_blocks = self._split_chapter_timing(block)
            for piece in chapter_blocks:
                if self._length_function(piece) > self._chunk_size:
                    chunks.extend(self.fine_splitter.split_text(piece))
                else:
                    chunks.append(piece)

        return [c.strip() for c in chunks if c and c.strip()]

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """切分 Document 并保留原 metadata（source/page 等）。"""
        output: list[Document] = []
        for doc in documents:
            for chunk in self.split_text(doc.page_content):
                output.append(Document(page_content=chunk, metadata=dict(doc.metadata or {})))
        return output

    @staticmethod
    def _split_verilog_blocks(text: str) -> list[str]:
        pattern = r"(module\s+[\w_]+[\s\S]*?endmodule)"
        parts = re.split(pattern, text, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p and p.strip()]

    @staticmethod
    def _is_verilog_block(text: str) -> bool:
        return bool(re.search(r"\bmodule\b|\bendmodule\b", text, flags=re.IGNORECASE))

    @staticmethod
    def _split_chapter_timing(text: str) -> list[str]:
        chapter_pattern = r"(?<=\n)(\d+(\.\d+)+[\u4e00-\u9fa5\s]*?[:：])"
        timing_pattern = r"(?<=\n)(图\d+[\u4e00-\u9fa5A-Za-z\s]*?(时序图|波形图|Timing Diagram))(?=\n)"

        timing_chunks = [c.strip() for c in re.split(timing_pattern, text) if c and c.strip()]

        out: list[str] = []
        for chunk in timing_chunks:
            out.extend([c.strip() for c in re.split(chapter_pattern, chunk) if c and c.strip()])
        return out or [text]
