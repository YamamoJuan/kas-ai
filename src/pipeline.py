from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .chunker import chunk_pages
from .citation import build_citations, Citation
from .document_loader import extract_pdf
from .embeddings import Embedder
from .llm import LLMClient
from .retriever import retrieve
from .vector_store import SearchResult, VectorStore


@dataclass
class PipelineResult:
    answer: str
    retrieved_chunks: List[SearchResult]
    citations: List[Citation]


class RAGPipeline:
    """Orkestrasi pipeline RAG: extract -> chunk -> embed -> FAISS -> retrieve -> LLM."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_pdf(self, pdf_path, document_name: str | None = None) -> tuple[int, int]:
        """Proses satu PDF. Return (jumlah_halaman_teks, jumlah_chunk)."""
        pages = extract_pdf(pdf_path, document_name=document_name)
        chunks = chunk_pages(pages, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise ValueError("Tidak ada chunk yang dihasilkan dari PDF.")
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.vector_store.add(embeddings, chunks)
        return len(pages), len(chunks)

    def query(self, question: str, llm: LLMClient, top_k: int = 5) -> PipelineResult:
        """Jawab pertanyaan dari seluruh dokumen yang sudah diproses."""
        if self.vector_store.count == 0:
            raise ValueError("Belum ada dokumen yang diproses. Silakan upload PDF terlebih dahulu.")

        results = retrieve(question, self.embedder, self.vector_store, top_k=top_k)
        if not results:
            return PipelineResult(
                answer="Tidak ada context relevan yang ditemukan dalam dokumen.",
                retrieved_chunks=[],
                citations=[],
            )

        chunks = [r.chunk for r in results]
        answer = llm.answer(question, chunks)
        citations = build_citations(chunks)
        return PipelineResult(answer=answer, retrieved_chunks=results, citations=citations)
