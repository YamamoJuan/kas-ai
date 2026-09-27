from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .chunker import Chunk


@dataclass
class Citation:
    document_name: str
    page_number: int
    chunk_id: str


def build_citations(chunks: List[Chunk], max_sources: int = 5) -> List[Citation]:
    """Bangun daftar citation unik dari chunk hasil retrieval.

    Sumber deduplikasi per (document_name, page_number) supaya UI tidak
    menampilkan halaman yang sama berulang kali. Urutan mengikuti urutan
    retrieval.
    """
    citations: List[Citation] = []
    seen = set()
    for chunk in chunks:
        key = (chunk.document_name, chunk.page_number)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                chunk_id=chunk.chunk_id,
            )
        )
        if len(citations) >= max_sources:
            break
    return citations


def format_citations(citations: List[Citation]) -> str:
    """Format citation menjadi teks untuk jawaban/UI."""
    if not citations:
        return "Tidak ada sumber ditemukan."
    lines = [f"- {c.document_name} — Page {c.page_number}" for c in citations]
    return "\n".join(lines)
