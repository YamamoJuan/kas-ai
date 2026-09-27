import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def _get(key: str, default: str = "") -> str:
    """Ambil config dari Streamlit Secrets (Cloud) dulu, lalu env/.env (local)."""
    try:
        import streamlit as st

        value = st.secrets.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(key, default)


LLM_API_KEY = _get("LLM_API_KEY", "")
LLM_BASE_URL = _get("LLM_BASE_URL", "https://api.gutsai.id/v1")
LLM_MODEL = _get("LLM_MODEL", "laguna-s2.1")
AVAILABLE_LLM_MODELS = [
    m.strip()
    for m in _get("AVAILABLE_LLM_MODELS", "nemotron-3-super,laguna-s2.1").split(",")
    if m.strip()
]

EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(_get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "100"))
TOP_K = int(_get("TOP_K", "5"))
