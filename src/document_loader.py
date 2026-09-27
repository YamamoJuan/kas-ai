from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pymupdf


@dataclass
class Page:
    """Teks hasil ekstraksi dari satu halaman PDF."""

    document_name: str
    page_number: int  # 1-based
    text: str


def extract_pdf(pdf_path: Path, document_name: str | None = None) -> List[Page]:
    """Ekstrak teks per halaman dari PDF dengan PyMuPDF.

    Halaman yang tidak punya teks (setelah trimming) diabaikan.
    """
    if document_name is None:
        document_name = pdf_path.name

    pages: List[Page] = []
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"Gagal membuka PDF: {exc}") from exc

    try:
        for index, page in enumerate(doc, start=1):
            raw = page.get_text("text")
            text = clean_text(raw)
            if text:
                pages.append(
                    Page(
                        document_name=document_name,
                        page_number=index,
                        text=text,
                    )
                )
    finally:
        doc.close()

    if not pages:
        raise ValueError(
            f"PDF '{document_name}' tidak memiliki teks yang bisa diproses. "
            "Kemungkinan dokumen adalah hasil scan murni (perlu OCR)."
        )
    return pages


def clean_text(text: str) -> str:
    """Bersihkan whitespace berlebihan tanpa mengubah makna."""
    text = text.replace("\u00a0", " ")
    lines = (" ".join(line.split()) for line in text.splitlines())
    cleaned = "\n".join(line for line in lines if line)
    return cleaned.strip()
