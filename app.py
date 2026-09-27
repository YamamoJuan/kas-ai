import tempfile
from pathlib import Path

import streamlit as st

from src.citation import format_citations
from src.config import (
    AVAILABLE_LLM_MODELS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
    TOP_K,
)
from src.embeddings import Embedder
from src.llm import LLMClient
from src.pipeline import RAGPipeline
from src.vector_store import VectorStore

st.set_page_config(page_title="Kas-AI", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Memuat embedding model...")
def load_pipeline(embedding_model: str, chunk_size: int, chunk_overlap: int) -> RAGPipeline:
    embedder = Embedder(embedding_model)
    return RAGPipeline(
        embedder=embedder,
        vector_store=VectorStore(embedder.dimension),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def init_state() -> None:
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("messages", [])


def render_sidebar():
    with st.sidebar:
        st.title("📄 Kas-AI")
        st.caption("Document RAG Assistant")

        st.subheader("Upload PDF")
        uploaded_files = st.file_uploader(
            "Pilih satu atau beberapa PDF",
            type=["pdf"],
            accept_multiple_files=True,
        )

        st.subheader("Uploaded documents")
        if st.session_state.documents:
            for doc in st.session_state.documents:
                st.success(f"{doc['name']} · {doc['pages']} hal · {doc['chunks']} chunk")
        else:
            st.info("Belum ada dokumen.")

        st.subheader("Settings")
        embedding_model = st.text_input("Embedding model", value=EMBEDDING_MODEL)
        default_idx = (
            AVAILABLE_LLM_MODELS.index(LLM_MODEL)
            if LLM_MODEL in AVAILABLE_LLM_MODELS
            else 0
        )
        llm_model = st.selectbox(
            "LLM model (Guts AI)",
            options=AVAILABLE_LLM_MODELS or [LLM_MODEL],
            index=default_idx,
        )
        top_k = st.slider("Top-K", 1, 20, TOP_K)
        st.caption(f"LLM base URL: `{LLM_BASE_URL or '-'}`")

        if st.button("Reset semua", use_container_width=True):
            st.session_state.clear()
            load_pipeline.clear()
            st.rerun()

    return uploaded_files, embedding_model, llm_model, top_k


def process_uploads(uploaded_files, pipeline: RAGPipeline) -> None:
    known = {doc["name"] for doc in st.session_state.documents}
    for uploaded in uploaded_files:
        if uploaded.name in known:
            continue

        tmp_path = None
        with st.status(f"Memproses {uploaded.name}...") as status:
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp_path = Path(tmp.name)

                page_count, chunk_count = pipeline.process_pdf(
                    tmp_path, document_name=uploaded.name
                )
                st.session_state.documents.append(
                    {"name": uploaded.name, "pages": page_count, "chunks": chunk_count}
                )
                status.update(
                    label=f"{uploaded.name}: {page_count} halaman, {chunk_count} chunk terindeks.",
                    state="complete",
                )
            except (ValueError, RuntimeError) as exc:
                status.update(label=f"{uploaded.name} gagal diproses.", state="error")
                st.error(str(exc))
            except Exception as exc:
                status.update(label=f"{uploaded.name} gagal diproses.", state="error")
                st.error(f"Error tidak terduga saat memproses {uploaded.name}: {exc}")
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)


def answer_question(question: str, pipeline: RAGPipeline, llm_model: str, top_k: int):
    if not st.session_state.documents:
        return "⚠️ Belum ada dokumen. Upload PDF terlebih dahulu di sidebar.", None
    try:
        llm = LLMClient(model=llm_model)
        with st.spinner("Mencari context & menyusun jawaban..."):
            result = pipeline.query(question, llm, top_k=top_k)
        citations = format_citations(result.citations) if result.citations else None
        return result.answer, citations
    except (ValueError, RuntimeError) as exc:
        return f"⚠️ {exc}", None
    except Exception as exc:
        return f"⚠️ Error tidak terduga: {exc}", None


def main() -> None:
    init_state()
    uploaded_files, embedding_model, llm_model, top_k = render_sidebar()

    st.title("Kas-AI: Document RAG Assistant")
    st.caption("Tanya apa pun tentang PDF Anda. Setiap jawaban menyertakan dokumen dan halaman sumber.")

    try:
        pipeline = load_pipeline(embedding_model, CHUNK_SIZE, CHUNK_OVERLAP)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    if uploaded_files:
        process_uploads(uploaded_files, pipeline)

    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources", expanded=True):
                    st.markdown(message["sources"])

    question = st.chat_input("Tanyakan sesuatu tentang dokumen Anda...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    answer, sources = answer_question(question, pipeline, llm_model, top_k)
    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("Sources", expanded=True):
                st.markdown(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


if __name__ == "__main__":
    main()
