# Kas-AI — Document RAG Assistant

**Kas-AI** is a document-based Retrieval-Augmented Generation (RAG) assistant that lets users upload PDF documents and ask questions about their content in natural language. Every answer is grounded in the uploaded documents and accompanied by verifiable citations: **document name + page number**.

Built with a clean, direct RAG pipeline — no LangChain, no LlamaIndex — so every stage is transparent, debuggable, and easy to extend.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Streamlit-1.64-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/FAISS-CPU-00ADD8?style=for-the-badge" alt="FAISS">
  <img src="https://img.shields.io/badge/Sentence%20Transformers-384d-FF6F00?style=for-the-badge" alt="Sentence Transformers">
  <img src="https://img.shields.io/badge/LLM-Guts%20AI-000000?style=for-the-badge" alt="Guts AI">
</p>

---

## Table of Contents

1. [What is Kas-AI?](#what-is-kas-ai)
2. [Problem & Solution](#problem--solution)
3. [Key Features](#key-features)
4. [How RAG Works in Kas-AI](#how-rag-works-in-kas-ai)
5. [System Architecture](#system-architecture)
6. [Tech Stack](#tech-stack)
7. [Why These Technical Choices?](#why-these-technical-choices)
8. [Installation](#installation)
9. [Environment Configuration](#environment-configuration)
10. [Running the Application](#running-the-application)
11. [Usage Guide](#usage-guide)
12. [API Abstraction](#api-abstraction)
13. [Error Handling](#error-handling)
14. [Testing & Verification](#testing--verification)
15. [End-to-End Demo Results](#end-to-end-demo-results)
16. [Project Structure](#project-structure)
17. [Limitations & Trade-offs](#limitations--trade-offs)
18. [Future Improvements](#future-improvements)

---

## What is Kas-AI?

Kas-AI turns static PDF documents into an interactive question-answering system. A user uploads one or more PDFs, and within seconds the system:

1. Extracts text page-by-page.
2. Splits the text into overlapping semantic chunks.
3. Converts each chunk into a vector embedding.
4. Stores the embeddings in a FAISS vector index.
5. Retrieves the most relevant chunks for each question.
6. Sends the retrieved context to an external LLM.
7. Returns an answer grounded in the document, with page-level citations.

This is a complete, working implementation of the RAG pattern, built for local execution with a production-ready architecture.

---

## Problem & Solution

### Problem

Reading long documents manually is slow. Generic LLMs are not trained on private/local files, so they either:

- Cannot access the document content at all, or
- Hallucinate answers instead of admitting the information is not in the document.

### Solution

Kas-AI injects relevant document chunks into the LLM prompt as context. The LLM is explicitly instructed to answer **only from the provided context**, which means:

- Answers are traceable to a real source chunk.
- Every answer includes the source document and page number.
- If the document does not contain the answer, the assistant says so instead of making things up.

---

## Key Features

- **Multi-PDF support** — upload multiple documents; retrieval searches across all of them.
- **Page-level citations** — sources are generated from real chunk metadata, never fabricated.
- **Configurable RAG pipeline** — chunk size, chunk overlap, embedding model, top-k, and LLM model are adjustable.
- **Provider-agnostic LLM abstraction** — any OpenAI-compatible API works by changing environment variables.
- **Automatic embedding model download** — the Sentence Transformer model is downloaded once and cached.
- **Clean modular codebase** — each RAG stage is isolated in its own module.
- **Clear error handling** — empty PDFs, missing API keys, failed API calls, empty retrieval, and more are handled with user-friendly messages.
- **Tested pipeline** — unit tests plus an end-to-end verification script.

---

## How RAG Works in Kas-AI

```text
                         ┌─────────────────────────────┐
                         │         PDF Upload          │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │   PDF Text Extraction       │
                         │   PyMuPDF (per page)        │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │      Text Cleaning          │
                         │  whitespace / blank lines   │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │         Chunking            │
                         │  size=500, overlap=100      │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │  Sentence Transformers      │
                         │  all-MiniLM-L6-v2 (384d)    │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │     FAISS Vector Store      │
                         │  IndexFlatIP + metadata map │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │        Retriever            │
                         │  query embedding → top-k    │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │    Prompt + Context         │
                         │  grounded system prompt     │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │      External LLM API       │
                         │  Guts AI (OpenAI-compatible)│
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │       Answer + Sources      │
                         │  document.pdf — Page N      │
                         └─────────────────────────────┘
```

---

## System Architecture

The project follows a modular, single-responsibility design. Each stage of the RAG pipeline lives in its own module.

```
┌──────────────────────────────────────────────────────────────────┐
│                         app.py (Streamlit UI)                    │
│  upload · chat · citations · settings · error handling           │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                       src/pipeline.py                            │
│               RAGPipeline — end-to-end orchestrator              │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────────┘
       ▼          ▼          ▼          ▼          ▼
┌────────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌────────────┐
│ document_  │ │ chunker  │ │ embed-   │ │ vector │ │ retriever  │
│ loader.py  │ │ .py      │ │ dings.py │ │ _store │ │ .py        │
│ PyMuPDF    │ │ chunking │ │ Sentence │ │ .py    │ │ top-k      │
│ per-page   │ │ metadata │ │ Transf.  │ │ FAISS  │ │ search     │
└────────────┘ └──────────┘ └──────────┘ └────────┘ └────────────┘
                                                         │
                                        ┌────────────────┘
                                        ▼
                              ┌──────────────────┐    ┌─────────────┐
                              │   prompt.py      │    │ citation.py │
                              │ grounded prompt  │    │ source      │
                              │ + context build  │    │ dedup       │
                              └────────┬─────────┘    └─────────────┘
                                       ▼
                              ┌──────────────────┐
                              │     llm.py       │
                              │ OpenAI-compatible│
                              │ client (Guts AI) │
                              └──────────────────┘
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `src/config.py` | Loads Streamlit Secrets (Cloud) or `.env` (local); centralizes all tunable parameters. |
| `src/document_loader.py` | Extracts text per PDF page with PyMuPDF; cleans whitespace; skips empty pages. |
| `src/chunker.py` | Splits text into overlapping chunks while preserving document name and page number. |
| `src/embeddings.py` | Wraps Sentence Transformers; lazy-loads and caches the model; normalizes vectors. |
| `src/vector_store.py` | FAISS `IndexFlatIP` index + in-memory mapping from vectors back to original chunks. |
| `src/retriever.py` | Embeds the query and runs top-k similarity search. |
| `src/prompt.py` | Builds the grounded system prompt and user context. |
| `src/llm.py` | Thin abstraction over any OpenAI-compatible API; includes retry and reasoning-model handling. |
| `src/citation.py` | Builds and formats citations from retrieved chunk metadata. |
| `src/pipeline.py` | Orchestrates the full RAG flow: index → retrieve → answer. |
| `app.py` | Streamlit frontend: file upload, chat, sources, model selection, error handling. |

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core language. |
| **Streamlit** | Rapid, clean web UI for the document assistant. |
| **PyMuPDF** | High-performance PDF text extraction with page-level access. |
| **Sentence Transformers** | Dense text embeddings (`all-MiniLM-L6-v2`, 384 dimensions). |
| **FAISS (CPU)** | Fast vector similarity search. |
| **OpenAI Python SDK** | Client for the OpenAI-compatible Guts AI API. |
| **Guts AI** | External LLM provider (models: `laguna-s2.1`, `nemotron-3-super`). |
| **python-dotenv** | Environment variable management. |
| **pytest** | Unit testing. |

---

## Why These Technical Choices?

### Direct RAG implementation (no LangChain / LlamaIndex)

The prompt explicitly prioritizes simplicity and maintainability. By implementing the pipeline directly, Kas-AI avoids framework overhead, hidden abstractions, and version churn. Every line of the pipeline is visible and debuggable.

### `all-MiniLM-L6-v2` for embeddings

- Lightweight (~80 MB) and fast on CPU.
- Produces compact 384-dimensional vectors.
- Strong general-purpose semantic search performance.
- Downloads automatically from Hugging Face and is cached locally.
- Easily replaceable via `EMBEDDING_MODEL` for multilingual workloads.

### FAISS `IndexFlatIP`

- Exact cosine similarity search (vectors are L2-normalized).
- Simple, deterministic, and perfect for small-to-medium local document collections.
- CPU-only, so it runs anywhere.

### OpenAI-compatible LLM abstraction

The application is not bound to a single provider. Any OpenAI-compatible endpoint can be used by setting `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY`. The current default is Guts AI, an Indonesian AI gateway with flat IDR pricing.

---

## Installation

### Prerequisites

- Python 3.10 or newer
- Internet connection for the first embedding model download and LLM API calls

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/kas-ai.git
cd kas-ai

# 2. Create a virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create the environment file
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env

# 5. Fill in your LLM credentials in .env
```

---

## Environment Configuration

Secrets are never hardcoded. Kas-AI loads them in this order:

1. **Streamlit Secrets** (`st.secrets`) — used on Streamlit Community Cloud.
2. **Environment / `.env`** — used for local development.

### Local: `.env`

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

```env
LLM_API_KEY=sk-guts-...
LLM_BASE_URL=https://api.gutsai.id/v1
LLM_MODEL=laguna-s2.1

# Optional overrides
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=5
```

### Streamlit Community Cloud: Secrets

`.env` is **not** available on Streamlit Cloud (and must never be committed). Configure Secrets instead:

1. Open your app on [share.streamlit.io](https://share.streamlit.io) → **Settings → Secrets**.
2. Paste the following (same format as `.streamlit/secrets.toml.example`):

```toml
LLM_API_KEY = "sk-guts-..."
LLM_BASE_URL = "https://api.gutsai.id/v1"
LLM_MODEL = "laguna-s2.1"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
```

3. Click **Save**, then **Reboot app**.

If `LLM_API_KEY` is missing in Secrets, the chat will show:

```text
⚠️ LLM_API_KEY belum diatur. ... Untuk Streamlit Cloud: isi Secrets (Settings → Secrets) ...
```

### Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | — | Secret key for the LLM provider (required). |
| `LLM_BASE_URL` | `https://api.gutsai.id/v1` | OpenAI-compatible base URL. |
| `LLM_MODEL` | `laguna-s2.1` | Default model shown in the UI. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence Transformers model name. |
| `CHUNK_SIZE` | `500` | Maximum chunk size in characters. |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks. |
| `TOP_K` | `5` | Number of chunks retrieved per question. |

> **Security note:** no API key is hardcoded in the source. Local `.env` and `.streamlit/secrets.toml` are gitignored. Only templates (`.env.example`, `.streamlit/secrets.toml.example`) are committed.

---

## Running the Application

### Local

```bash
streamlit run app.py
```

Streamlit will open a browser tab automatically. If not, visit:

```text
http://localhost:8501
```

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new app on [share.streamlit.io](https://share.streamlit.io) pointing to `app.py`.
3. Fill **Settings → Secrets** as shown above.
4. Wait for the first boot (embedding model download can take 1–2 minutes).

---

## Usage Guide

### 1. Upload documents

In the sidebar, upload one or more PDF files. Each document is processed with a visible status indicator:

```text
laporan.pdf: 12 halaman, 14 chunk terindeks.
```

### 2. Configure the assistant (optional)

The sidebar allows you to:

- Select the **LLM model** (`laguna-s2.1` or `nemotron-3-super`).
- Adjust **Top-K** retrieval depth.
- Override the **embedding model** if needed.

### 3. Ask questions

Type a question in the chat input. The assistant responds with:

- A grounded answer.
- A **Sources** expander listing `document.pdf — Page N`.

Example output:

```text
User:
Siapa manajer proyek Nebula Gate?

Kas-AI:
Manajer proyek Nebula Gate adalah Dr. Siti Rahmawati.

Sources:
- nebula_gate.pdf — Page 1
```

### 4. Reset

Use the **Reset semua** button in the sidebar to clear the indexed documents and chat history.

---

## API Abstraction

`src/llm.py` provides a minimal, reusable LLM client:

```python
from src.llm import LLMClient

llm = LLMClient(model="laguna-s2.1")
answer = llm.answer(question, retrieved_chunks)
```

Features:

- Validates that `LLM_API_KEY` and `LLM_MODEL` are configured.
- Accepts only a user question and retrieved context; never fabricates its own context.
- Retries transient failures (up to 3 attempts).
- Handles reasoning models that return content in `reasoning` rather than `content`.
- Raises descriptive, user-friendly errors on failure.

The system prompt enforces these rules:

1. Answer **only** from the provided context.
2. Do not use external knowledge.
3. If the answer is not in the context, say so explicitly.
4. Use available sources to support the answer.
5. Respond clearly and naturally.

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Empty PDF / scanned PDF without text | User sees a clear message explaining OCR is required. |
| Missing `LLM_API_KEY` | User is told to fill `.env` (local) or Streamlit Secrets (Cloud). |
| LLM API failure | User sees a descriptive error with the underlying cause. |
| Embedding model failure | User sees the model name and likely cause. |
| FAISS error | User sees a clear runtime error message. |
| Question before document upload | User is prompted to upload a PDF first. |
| Empty retrieval result | User is told that no relevant context was found in the documents. |

All errors surface in the Streamlit UI as clear messages — no silent failures.

---

## Testing & Verification

### Unit Tests

```bash
pytest -q
```

Current result:

```text
5 passed
```

Tests cover:

- PDF extraction and page-number preservation.
- Text cleaning.
- Chunking with metadata preservation.
- Citation generation and deduplication.
- FAISS retrieval relevance.

### End-to-End Verification

The script below runs the full pipeline (PDF → extract → chunk → embed → FAISS → retrieve → citation) without needing an LLM key:

```bash
python scripts/verify_e2e.py
```

### Full App Flow (including LLM)

This script generates a realistic PDF, indexes it, and sends multiple questions through the complete RAG pipeline with the configured Guts AI model:

```bash
python scripts/test_app_flow.py
```

---

## End-to-End Demo Results

The following results were produced by running `scripts/test_app_flow.py` with a generated 3-page PDF about a fictional project called **Nebula Gate**.

**Model:** Guts AI `laguna-s2.1` · **Embeddings:** `all-MiniLM-L6-v2` · **Top-K:** 5

| Question | Answer | Citation |
|---|---|---|
| Kapan dan di mana proyek Nebula Gate diluncurkan? | Proyek Nebula Gate diluncurkan pada tahun **2019** di kota **Bandung**. | `nebula_gate.pdf — Page 1` |
| Siapa manajer proyek Nebula Gate? | Manajer proyek Nebula Gate adalah **Dr. Siti Rahmawati**. | `nebula_gate.pdf — Page 1` |
| Berapa target pengguna tahun pertama? | Target pengguna tahun pertama adalah **50.000 orang**. | `nebula_gate.pdf — Page 2` |
| Siapa mitra utama Nebula Gate? | Mitra utama adalah **Bank Nusantara** dan **PT Digital Nusantara**. | `nebula_gate.pdf — Page 3` |

**Result: 4/4 questions answered correctly with real page-level citations.**

---

## Project Structure

```text
kas-ai/
├── app.py                    # Streamlit UI entry point
├── src/
│   ├── __init__.py
│   ├── config.py             # Environment & parameter configuration
│   ├── document_loader.py    # PDF extraction (PyMuPDF) + text cleaning
│   ├── chunker.py            # Configurable chunking with overlap
│   ├── embeddings.py         # Sentence Transformers wrapper
│   ├── vector_store.py       # FAISS index + metadata mapping
│   ├── retriever.py          # Query embedding → top-k search
│   ├── llm.py                # OpenAI-compatible LLM client
│   ├── prompt.py             # Grounded RAG prompt + context builder
│   ├── citation.py           # Citation generation & formatting
│   └── pipeline.py           # End-to-end RAG orchestrator
├── data/
│   └── .gitkeep              # Local data directory (gitignored content)
├── scripts/
│   ├── verify_e2e.py         # Pipeline verification (no LLM key required)
│   └── test_app_flow.py      # Full app flow test with LLM
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py      # Unit tests
├── requirements.txt
├── .env.example                       # Local env template (safe to commit)
├── .streamlit/
│   └── secrets.toml.example           # Streamlit Cloud Secrets template
├── .gitignore                         # Excludes .env, secrets.toml, venv, caches
└── README.md
```

---

## Limitations & Trade-offs

- **Image-only/scanned PDFs** are not supported; they require OCR, which is out of scope for this version.
- **In-memory FAISS index** — indexed documents are lost when the app restarts. Persistence is a planned improvement.
- **Character-based chunking** — simple and reliable, though token-based chunking could be more precise for some LLMs.
- **English-biased embedding model** — `all-MiniLM-L6-v2` works well generally but can be replaced with a multilingual model for Indonesian-heavy corpora.
- **No streaming** — LLM answers are displayed after completion rather than token-by-token.

---

## Future Improvements

- Persist FAISS index and chunk metadata to disk.
- OCR support for scanned PDFs.
- Hybrid retrieval (dense + BM25) for better keyword precision.
- Streaming LLM responses.
- Retrieval evaluation metrics (hit-rate, faithfulness, RAGAS).
- Conversational memory (multi-turn follow-up questions).
- Docker deployment.

---

## Credits

- [Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for embeddings.
- [FAISS](https://github.com/facebookresearch/faiss) for vector search.
- [Guts AI](https://gutsai.id) for LLM inference.
- [Streamlit](https://streamlit.io) for the UI framework.
- [PyMuPDF](https://pymupdf.io) for PDF processing.
