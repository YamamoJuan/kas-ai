from __future__ import annotations

from typing import List

from .embeddings import Embedder
from .vector_store import SearchResult, VectorStore


def retrieve(
    query: str,
    embedder: Embedder,
    vector_store: VectorStore,
    top_k: int = 5,
) -> List[SearchResult]:
    """Cari top-k chunk relevan dan kembalikan beserta metadata aslinya."""
    if not query.strip():
        raise ValueError("Pertanyaan tidak boleh kosong")
    if vector_store.count == 0:
        return []

    query_embedding = embedder.embed_query(query.strip())
    return vector_store.search(query_embedding, top_k=top_k)
