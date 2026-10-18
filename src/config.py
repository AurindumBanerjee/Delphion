"""Central configuration. Everything is env-driven so the same code runs
locally, on a Hugging Face Space, or against a local Ollama instance."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Must run before Settings() reads os.environ below. (It used to be called in
# app.py *after* this module was imported, so a local .env was silently ignored.)
load_dotenv()


@dataclass(frozen=True)
class Settings:
    # --- Embeddings (runs in-process on CPU, no API key, no cost) ---
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

    # --- Generation (any OpenAI-compatible endpoint) ---
    # Groq / OpenRouter / Together / Google AI Studio / local Ollama all work.
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    # --- Retrieval ---
    default_k: int = int(os.getenv("DEFAULT_K", "4"))
    # dense | bm25 | hybrid | hybrid_rerank  (see src/retrieval.py)
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid_rerank")
    rerank_model: str = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "20"))

    @property
    def llm_configured(self) -> bool:
        # Local servers (Ollama, vLLM) need no key; hosted ones do.
        key = self.llm_api_key.strip()
        has_key = bool(key) and key.lower() not in {"your-key-here", "changeme"}
        return has_key or "localhost" in self.llm_base_url or "127.0.0.1" in self.llm_base_url


settings = Settings()

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise document analyst. Answer the user's question using ONLY the "
    "numbered context passages provided. Cite the passages you used inline as [1], [2], etc. "
    "If the context does not contain the answer, say exactly: "
    "'I could not find the answer in the provided documents.' Do not use outside knowledge."
)
