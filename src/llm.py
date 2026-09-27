from __future__ import annotations

import time
from typing import List

from openai import OpenAI

from .chunker import Chunk
from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .prompt import build_context, build_messages


class LLMClient:
    """Abstraksi tipis untuk LLM OpenAI-compatible (Guts AI, dll.)."""

    def __init__(
        self,
        api_key: str = LLM_API_KEY,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
        max_retries: int = 3,
    ):
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or "https://api.gutsai.id/v1").strip()
        self.model = (model or "").strip()
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError(
                "LLM_API_KEY belum diatur. Salin .env.example menjadi .env lalu isi API key."
            )
        if not self.model:
            raise ValueError("LLM_MODEL belum diatur di .env")

        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as exc:
            raise RuntimeError(f"Gagal membuat LLM client: {exc}") from exc

    def answer(self, question: str, chunks: List[Chunk], max_tokens: int = 700) -> str:
        if not question.strip():
            raise ValueError("Pertanyaan tidak boleh kosong")

        context = build_context(chunks)
        messages = build_messages(question.strip(), context)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_error = RuntimeError(
                    "Gagal memanggil LLM API. Periksa LLM_API_KEY, LLM_BASE_URL, "
                    f"LLM_MODEL, dan koneksi internet. Detail: {exc}"
                )
                if attempt < self.max_retries:
                    time.sleep(attempt)
                    continue
                raise last_error from exc

            content = self._extract_content(response)
            if content:
                return content

            last_error = RuntimeError(
                "LLM mengembalikan jawaban kosong atau format respons tidak valid."
            )
            if attempt < self.max_retries:
                time.sleep(attempt)

        raise last_error or RuntimeError("LLM gagal menghasilkan jawaban.")

    @staticmethod
    def _extract_content(response) -> str | None:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None

        message = getattr(choices[0], "message", None)
        if message is None:
            return None

        content = getattr(message, "content", None)
        if content and str(content).strip():
            return str(content).strip()

        # Reasoning model kadang menaruh jawaban di field reasoning.
        reasoning = getattr(message, "reasoning", None)
        if reasoning and str(reasoning).strip():
            return str(reasoning).strip()
        return None
