from __future__ import annotations

from sentence_transformers import SentenceTransformer


class Embedder:
    """Wrapper Sentence Transformers untuk embedding chunk & query."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Gagal memuat embedding model '{self.model_name}'. "
                    f"Pastikan koneksi internet tersedia saat download pertama. Detail: {exc}"
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]):
        if not texts:
            return []
        try:
            return self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Gagal membuat embeddings: {exc}") from exc

    def embed_query(self, text: str):
        return self.embed_documents([text])[0]
