from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from src.chunker import chunk_pages
from src.citation import build_citations, format_citations
from src.document_loader import Page, clean_text, extract_pdf
from src.embeddings import Embedder
from src.retriever import retrieve
from src.vector_store import VectorStore


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "Kas-AI adalah asisten dokumen berbasis RAG. Tujuan utamanya membantu "
        "pengguna bertanya pada isi PDF dengan jawaban yang dilengkapi citation.",
    )
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "Pipeline Kas-AI: ekstraksi teks, chunking, embedding Sentence Transformers, "
        "pencarian FAISS, lalu jawaban dari LLM eksternal.",
    )
    page3 = doc.new_page()  # empty page — harus diabaikan
    doc.save(path)
    doc.close()
    return path


def test_extract_pdf_keeps_page_numbers(sample_pdf: Path):
    pages = extract_pdf(sample_pdf, document_name="sample.pdf")
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert "Kas-AI" in pages[0].text
    assert "FAISS" in pages[1].text


def test_clean_text_collapses_whitespace():
    assert clean_text("  hello   world \n\n  ") == "hello world"


def test_chunking_preserves_metadata():
    pages = [
        Page(document_name="doc.pdf", page_number=3, text="alpha " * 200),
        Page(document_name="doc.pdf", page_number=4, text="beta " * 50),
    ]
    chunks = chunk_pages(pages, chunk_size=120, chunk_overlap=20)
    assert chunks
    assert all(c.document_name == "doc.pdf" for c in chunks)
    assert {c.page_number for c in chunks} == {3, 4}
    assert all(c.chunk_id for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_citation_generation_dedupes_pages():
    from src.chunker import Chunk

    chunks = [
        Chunk("a.pdf", 1, "a-p1-c1", "teks satu"),
        Chunk("a.pdf", 1, "a-p1-c2", "teks dua"),
        Chunk("a.pdf", 2, "a-p2-c1", "teks tiga"),
        Chunk("b.pdf", 5, "b-p5-c1", "teks empat"),
    ]
    citations = build_citations(chunks)
    assert [(c.document_name, c.page_number) for c in citations] == [
        ("a.pdf", 1),
        ("a.pdf", 2),
        ("b.pdf", 5),
    ]
    formatted = format_citations(citations)
    assert "a.pdf — Page 1" in formatted
    assert "b.pdf — Page 5" in formatted


def test_retrieval_returns_relevant_chunk(sample_pdf: Path):
    pages = extract_pdf(sample_pdf, document_name="sample.pdf")
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=50)
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    store = VectorStore(embedder.dimension)
    store.add(embedder.embed_documents([c.text for c in chunks]), chunks)

    results = retrieve("Apa tujuan utama Kas-AI?", embedder, store, top_k=2)
    assert results
    assert any("tujuan" in r.chunk.text.lower() or "kas-ai" in r.chunk.text.lower() for r in results)
    assert all(r.chunk.page_number >= 1 for r in results)
