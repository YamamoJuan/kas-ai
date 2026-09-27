from __future__ import annotations

from .chunker import Chunk

SYSTEM_PROMPT = """Kamu adalah Kas-AI, asisten tanya-jawab dokumen (RAG).
Aturan utama:
1. Jawablah HANYA berdasarkan context yang diberikan.
2. Jangan mengarang informasi atau menggunakan pengetahuan eksternal.
3. Jika jawaban tidak ditemukan dalam context, katakan secara jelas bahwa informasi tersebut tidak ditemukan di dokumen yang diunggah.
4. Gunakan source yang tersedia untuk mendukung jawaban.
5. Jawaban harus jelas, natural, dan gunakan Bahasa Indonesia kecuali user bertanya dalam bahasa lain.
6. Jangan menyebutkan label teknis seperti chunk_id/vector/F.A.Q internal; fokus pada isi dokumen."""


def build_context(chunks: list[Chunk]) -> str:
    """Susun context untuk LLM lengkap dengan penanda sumber halaman."""
    if not chunks:
        return "(Tidak ada context)"

    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] Sumber: {chunk.document_name} - Halaman {chunk.page_number}\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, context: str) -> list[dict]:
    """Bangun daftar pesan OpenAI-compatible (system + user)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context dokumen:\n\"\"\"\n{context}\n\"\"\"\n\n"
                f"Pertanyaan user:\n{question}"
            ),
        },
    ]
