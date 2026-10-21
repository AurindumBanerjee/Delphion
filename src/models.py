"""Model providers.

Two seams, so switching vendors is a config change and never a code change:
  get_embeddings() -> a LangChain Embeddings object (local, CPU, free)
  chat()           -> one call against any OpenAI-compatible chat endpoint
"""

from contextvars import ContextVar
from functools import lru_cache
from typing import Callable, Iterator, List, Mapping, Optional

from langchain_core.embeddings import Embeddings

from src.config import settings

# An optional per-request hook so the UI can show real embedding progress. It is set and
# read inside one request's context, so concurrent visitors never see each other's ticks.
_embed_progress: ContextVar[Optional[Callable[[int], None]]] = ContextVar(
    "embed_progress", default=None
)
_TICK_EVERY = 16  # report every N embedded passages, not every one


def set_embed_progress(callback: Optional[Callable[[int], None]]):
    """Install `callback(n_just_embedded)` for the current request; returns a reset token."""
    return _embed_progress.set(callback)


def reset_embed_progress(token) -> None:
    _embed_progress.reset(token)


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
        out: List[List[float]] = []
        tick = _embed_progress.get()
        for vec in self._model.embed(list(texts)):
            out.append(vec.tolist())
            if tick and len(out) % _TICK_EVERY == 0:
                tick(_TICK_EVERY)
        if tick and len(out) % _TICK_EVERY:
            tick(len(out) % _TICK_EVERY)
        return out

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


def _make_client(api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=settings.llm_base_url, api_key=api_key)


@lru_cache(maxsize=1)
def _client():
    """The shared server-key client. Cached: it is the same for every visitor."""
    return _make_client(settings.llm_api_key or "not-needed")


def _client_for(api_key: Optional[str]):
    """A visitor's own key gets a fresh client per call. It is never cached and never
    stored in `settings`, so it cannot outlive the request or reach another session."""
    return _make_client(api_key) if api_key else _client()


def _messages(system_prompt: str, user_prompt: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def chat(system_prompt: str, user_prompt: str, api_key: Optional[str] = None,
         on_headers: Optional[Callable[[Mapping], None]] = None) -> str:
    """Single-turn completion against the configured endpoint."""
    raw = _client_for(api_key).chat.completions.with_raw_response.create(
        model=settings.llm_model,
        temperature=settings.temperature,
        messages=_messages(system_prompt, user_prompt),
    )
    if on_headers:
        on_headers(raw.headers)
    return raw.parse().choices[0].message.content


def chat_stream(system_prompt: str, user_prompt: str, api_key: Optional[str] = None,
                on_headers: Optional[Callable[[Mapping], None]] = None) -> Iterator[str]:
    """Streaming variant: yields text deltas as the model produces them. The rate-limit
    headers arrive with the response head, so `on_headers` fires before the first token."""
    raw = _client_for(api_key).chat.completions.with_raw_response.create(
        model=settings.llm_model,
        temperature=settings.temperature,
        stream=True,
        messages=_messages(system_prompt, user_prompt),
    )
    if on_headers:
        on_headers(raw.headers)
    for event in raw.parse():
        if event.choices and event.choices[0].delta and event.choices[0].delta.content:
            yield event.choices[0].delta.content


def probe(api_key: Optional[str] = None) -> Mapping:
    """One minimal request (max_tokens=1); returns the response headers. Used to validate
    a key and to refresh the usage panel. Costs one request."""
    raw = _client_for(api_key).chat.completions.with_raw_response.create(
        model=settings.llm_model,
        max_tokens=1,
        messages=[{"role": "user", "content": "ping"}],
    )
    return raw.headers
