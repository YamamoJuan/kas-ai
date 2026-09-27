from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .document_loader import Page


@dataclass
class Chunk:
    document_name: str
    page_number: int
    chunk_id: str
    text: str


def chunk_pages(
    pages: List[Page],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Chunk]:
    """Pecah halaman menjadi chunk berukuran `chunk_size` karakter.

    Batas chunk berdasarkan paragraf (`\\n`) agar lebih natural; bila satu
    paragraf lebih panjang dari `chunk_size`, fallback ke pemotongan per
    kata. Setiap chunk tetap membawa metadata halaman.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size harus lebih besar dari 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap tidak boleh negatif")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap harus lebih kecil dari chunk_size")

    chunks: List[Chunk] = []
    for page in pages:
        pieces = _split_text(page.text, chunk_size, chunk_overlap)
        for piece in pieces:
            chunks.append(
                Chunk(
                    document_name=page.document_name,
                    page_number=page.page_number,
                    chunk_id=f"{page.document_name}-p{page.page_number}-c{len(chunks) + 1}",
                    text=piece,
                )
            )
    return chunks


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split teks per paragraf dengan overlap; fallback per kata."""
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if paragraphs:
        pieces = _split_by_units(paragraphs, "\n", chunk_size, chunk_overlap)
        if pieces:
            return pieces

    words = text.split()
    return _split_by_units(words, " ", chunk_size, chunk_overlap)


def _split_by_units(
    units: List[str],
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Gabungkan unit (paragraf/kata) sampai mendekati chunk_size."""
    pieces: List[str] = []
    buffer: List[str] = []
    current_len = 0

    for unit in units:
        addition = len(unit) + (len(separator) if buffer else 0)
        if current_len + addition > chunk_size and buffer:
            pieces.append(separator.join(buffer))
            overlap_buffer: List[str] = []
            overlap_len = 0
            for prev in reversed(buffer):
                needed = chunk_overlap - overlap_len
                if needed <= 0:
                    break
                token = prev[:needed]
                overlap_buffer.append(token)
                overlap_len += len(token) + (len(separator) if overlap_len > 0 else 0)
            buffer = list(reversed(overlap_buffer))
            current_len = sum(len(b) for b in buffer) + len(separator) * max(0, len(buffer) - 1)

        buffer.append(unit)
        current_len += len(unit) + (len(separator) if len(buffer) > 1 else 0)

    if buffer:
        pieces.append(separator.join(buffer))
    return pieces
