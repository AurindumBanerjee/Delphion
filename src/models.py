"""Model providers.

Two seams, so switching vendors is a config change and never a code change:
  get_embeddings() -> a LangChain Embeddings object (local, CPU, free)
  chat()           -> one call against any OpenAI-compatible chat endpoint
"""

from functools import lru_cache
from typing import Iterator, List

from langchain_core.embeddings import Embeddings

from src.config import settings


class FastEmbedEmbeddings(Embeddings):
    """Runs an ONNX sentence-embedding model in-process.

    Chosen over sentence-transformers/torch deliberately: ~5x smaller install
    and a much faster cold start, which matters on a free 2-vCPU Space.
    """

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Cached: the model is loaded once per process, not once per upload."""
    return FastEmbedEmbeddings(settings.embed_model)


class FastEmbedReranker:
    """Cross-encoder reranker (ONNX, CPU). Reads query and passage together,
    so it judges relevance far better than comparing two separate vectors.
    ms-marco-MiniLM-L-6 is ~80MB and scores 20 pairs in well under a second."""

    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        return list(self._model.rerank(query, documents))


@lru_cache(maxsize=1)
def get_reranker():
    """Loaded once per process. Returns None (and the app degrades to plain
    hybrid retrieval) if the model cannot be loaded."""
    try:
        return FastEmbedReranker(settings.rerank_model)
    except Exception as exc:  # pragma: no cover - network/model availability
        from loguru import logger

        logger.warning(f"Reranker unavailable ({exc}); falling back to hybrid without rerank")
        return None


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    return OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "not-needed",
    )


def chat(system_prompt: str, user_prompt: str) -> str:
    """Single-turn completion against the configured endpoint."""
    completion = _client().chat.completions.create(
        model=settings.llm_model,
        temperature=settings.temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


def chat_stream(system_prompt: str, user_prompt: str) -> Iterator[str]:
    """Streaming variant: yields text deltas as the model produces them."""
    stream = _client().chat.completions.create(
        model=settings.llm_model,
        temperature=settings.temperature,
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    for event in stream:
        if event.choices and event.choices[0].delta and event.choices[0].delta.content:
            yield event.choices[0].delta.content
