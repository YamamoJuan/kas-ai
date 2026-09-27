"""Verifikasi end-to-end pipeline tanpa memanggil LLM API.

Menjalankan: PDF → extract → chunk → embed → FAISS → retrieve → citation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf

from src.citation import format_citations, build_citations
from src.chunker import chunk_pages
from src.document_loader import extract_pdf
from src.embeddings import Embedder
from src.pipeline import RAGPipeline
from src.prompt import build_context, build_messages
from src.vector_store import VectorStore


def make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text(
        (72, 72),
        "Kas-AI adalah asisten dokumen berbasis RAG. Tujuan utama sistem ini "
        "adalah menjawab pertanyaan pengguna berdasarkan isi dokumen PDF.",
    )
    p2 = doc.new_page()
    p2.insert_text(
        (72, 72),
        "Komponen utama: PyMuPDF untuk ekstraksi, Sentence Transformers untuk "
        "embedding, FAISS untuk vector search, dan LLM eksternal untuk jawaban.",
    )
    doc.save(path)
    doc.close()


def main() -> None:
    pdf = Path("data") / "verify_sample.pdf"
    make_pdf(pdf)

    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=VectorStore(embedder.dimension),
        chunk_size=500,
        chunk_overlap=100,
    )
    pages, chunks = pipeline.process_pdf(pdf, document_name="verify_sample.pdf")
    assert pages == 2 and chunks >= 1, (pages, chunks)

    results = pipeline.vector_store.search(
        embedder.embed_query("Apa tujuan utama Kas-AI?"),
        top_k=3,
    )
    assert results, "retrieval kosong"
    assert any(r.chunk.page_number == 1 for r in results)

    retrieved_chunks = [r.chunk for r in results]
    citations = build_citations(retrieved_chunks)
    assert citations
    assert "verify_sample.pdf — Page" in format_citations(citations)

    context = build_context(retrieved_chunks)
    messages = build_messages("Apa tujuan utama Kas-AI?", context)
    assert messages[0]["role"] == "system"
    assert "Context dokumen" in messages[1]["content"]

    print("E2E OK")
    print(f"  pages={pages} chunks={chunks}")
    print(f"  top result page={results[0].chunk.page_number} score={results[0].score:.3f}")
    print("  sources:")
    print(format_citations(citations))
    pdf.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
