"""Tes end-to-end jalur yang dipakai app.py: PDF random → index → chat → answer + sources."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf

from src.citation import format_citations
from src.config import AVAILABLE_LLM_MODELS, EMBEDDING_MODEL, LLM_MODEL, TOP_K
from src.embeddings import Embedder
from src.llm import LLMClient
from src.pipeline import RAGPipeline
from src.vector_store import VectorStore


def make_random_pdf(path: Path) -> None:
    """PDF 'random' berisi fakta unik agar jawaban bisa diverifikasi."""
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_textbox(
        pymupdf.Rect(72, 72, 520, 720),
        "Proyek Nebula Gate diluncurkan tahun 2019 di kota Bandung. "
        "Anggaran awal proyek adalah Rp 12,5 miliar. "
        "Manajer proyek: Dr. Siti Rahmawati.",
        fontsize=12,
    )
    p2 = doc.new_page()
    p2.insert_textbox(
        pymupdf.Rect(72, 72, 520, 720),
        "Fitur utama Nebula Gate adalah sistem antrean digital QRIS dan "
        "monitoring real-time. Target pengguna tahun pertama: 50.000 orang. "
        "Kantor pusat terletak di Jalan Asia Afrika No. 88.",
        fontsize=12,
    )
    p3 = doc.new_page()
    p3.insert_textbox(
        pymupdf.Rect(72, 72, 520, 720),
        "Pada 2023, Nebula Gate mencatat 120.000 transaksi. "
        "Mitra utama: Bank Nusantara dan PT Digital Nusantara.",
        fontsize=12,
    )
    doc.save(path)
    doc.close()


def main() -> None:
    pdf = ROOT / "data" / "nebula_gate.pdf"
    make_random_pdf(pdf)
    print(f"[1] PDF dibuat: {pdf.name}")

    print(f"[2] Load embedder: {EMBEDDING_MODEL}")
    embedder = Embedder(EMBEDDING_MODEL)
    pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=VectorStore(embedder.dimension),
        chunk_size=500,
        chunk_overlap=100,
    )

    pages, chunks = pipeline.process_pdf(pdf, document_name=pdf.name)
    print(f"[3] Indexed: {pages} halaman, {chunks} chunk")

    questions = [
        "Kapan dan di mana proyek Nebula Gate diluncurkan?",
        "Siapa manajer proyek Nebula Gate?",
        "Berapa target pengguna tahun pertama?",
        "Siapa mitra utama Nebula Gate?",
    ]

    model = LLM_MODEL if LLM_MODEL in AVAILABLE_LLM_MODELS else AVAILABLE_LLM_MODELS[0]
    print(f"[4] Chat via Guts AI model={model} top_k={TOP_K}")
    llm = LLMClient(model=model)

    ok = 0
    for i, q in enumerate(questions, 1):
        result = pipeline.query(q, llm, top_k=TOP_K)
        sources = format_citations(result.citations)
        print(f"\n--- Q{i}: {q}")
        print(f"A: {result.answer}")
        print(f"Sources:\n{sources}")
        if result.answer and result.citations:
            ok += 1

    pdf.unlink(missing_ok=True)
    print(f"\n[DONE] {ok}/{len(questions)} pertanyaan mendapat jawaban + sources")
    if ok != len(questions):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
