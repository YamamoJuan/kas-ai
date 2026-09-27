from __future__ import annotations

from dataclasses import dataclass
from typing import List

import faiss
import numpy as np

from .chunker import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    """Vector store sederhana berbasis FAISS + metadata chunk in-memory."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # cosine similarity untuk normalized vectors
        self.chunks: List[Chunk] = []

    @property
    def count(self) -> int:
        return len(self.chunks)

    def add(self, embeddings, chunks: List[Chunk]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("Jumlah embeddings dan chunks tidak sama")

        vectors = _to_float32_matrix(embeddings)
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Dimensi embedding {vectors.shape[1]} tidak cocok dengan index {self.dimension}"
            )
        try:
            self.index.add(vectors)
        except Exception as exc:
            raise RuntimeError(f"Gagal menambahkan vector ke FAISS: {exc}") from exc
        self.chunks.extend(chunks)

    def search(self, query_embedding, top_k: int = 5) -> List[SearchResult]:
        if self.count == 0:
            return []
        if top_k <= 0:
            return []

        query = _to_float32_matrix([query_embedding])
        effective_k = min(top_k, self.count)
        try:
            distances, indices = self.index.search(query, effective_k)
        except Exception as exc:
            raise RuntimeError(f"FAISS similarity search gagal: {exc}") from exc

        results: List[SearchResult] = []
        for idx, score in zip(indices[0], distances[0]):
            if 0 <= int(idx) < self.count:
                results.append(SearchResult(chunk=self.chunks[int(idx)], score=float(score)))
        return results


def _to_float32_matrix(vectors) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("Embeddings harus berbentuk array 1D/2D")
    return arr
