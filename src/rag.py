"""The RAG loop: retrieve -> build a cited prompt -> generate (optionally streamed)."""

import re
from typing import Iterator, List, Optional, Tuple

from langchain_core.documents import Document
from loguru import logger
from openai import APIError, AuthenticationError, PermissionDeniedError, RateLimitError

from src.config import DEFAULT_SYSTEM_PROMPT, settings
from src.models import chat, chat_stream
from src.usage import fmt_duration, limit_kind, retry_after_seconds


def retrieve(store, question: str, k: int, mode: Optional[str] = None) -> List[Document]:
    mode = mode or settings.retrieval_mode
    if hasattr(store, "search"):
        return store.search(question, k=k, mode=mode)
    return store.similarity_search(question, k=k)  # plain vector store


def format_context(docs: List[Document]) -> str:
    """Number each passage so the model can cite it as [n]."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{i}] ({src}, p.{page})\n{doc.page_content}")
    return "\n\n".join(blocks)


def format_sources(docs: List[Document]) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        preview = " ".join(doc.page_content.split())[:180]
        lines.append(f"**[{i}] {src} — page {page}**\n\n> {preview}…")
    return "\n\n".join(lines) if lines else "_No passages retrieved._"


NO_LLM = (
    "**No LLM endpoint configured.** Retrieval worked — the passages are shown "
    "below — but generation needs `LLM_API_KEY` (and optionally `LLM_BASE_URL` "
    "and `LLM_MODEL`) to be set, or your own key under \"Use your own API key\"."
)


def _normalise_citations(text: str) -> str:
    """Some models cite as 【2】 or [2†source]; show them all as plain [2]."""
    text = re.sub(r"【(\d+)(?:†[^】]*)?】", r"[\1]", text)
    return text.replace(" ", " ")


def _redact(text: str, *keys: Optional[str]) -> str:
    """Last line of defence: no key, shared or personal, may appear in UI text."""
    for key in keys:
        if key and len(key) >= 4:
            text = text.replace(key, "[redacted]")
    return text


def _limit_message(kind: Optional[str], retry_after: Optional[float], own: bool) -> str:
    label = {"day": "Daily", "minute": "Per-minute"}.get(kind, "Rate")
    when = f", resets in {fmt_duration(retry_after)}" if retry_after else ", try again later"
    if own:
        return (
            f"**{label} limit reached{when}.** This is your own key's limit, not the demo's "
            "quota. Retrieval still works: the passages are shown below. You can wait, or "
            "remove your key to use the shared demo key."
        )
    return (
        f"**{label} limit reached{when}.** The shared demo key is used by all visitors of "
        "this Space. Retrieval still works: the passages are shown below. You can try again "
        "later, or use your own key under \"Use your own API key\"."
    )


def _bad_key_message(own: bool) -> str:
    if own:
        return (
            "**Your key was rejected by the provider.** Retrieval still works: the passages "
            "are shown below. Remove your key to use the shared demo key."
        )
    return (
        "**The shared demo key was rejected.** The Space owner needs to check the "
        "`LLM_API_KEY` secret. Retrieval still works: the passages are shown below."
    )


def _error_note(exc: Exception, api_key: Optional[str], usage) -> str:
    """Map a generation failure to UI text. Provider error bodies are never shown."""
    own = bool(api_key)
    if isinstance(exc, RateLimitError):
        retry = retry_after_seconds(getattr(getattr(exc, "response", None), "headers", None))
        kind = limit_kind(exc)
        if usage is not None:
            usage.on_limit(kind, retry)
        return _limit_message(kind, retry, own)
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return _bad_key_message(own)
    if isinstance(exc, APIError):
        return f"Generation failed ({type(exc).__name__}). Please try again."
    shared_key = getattr(settings, "llm_api_key", "")
    return f"_Generation failed: {_redact(str(exc), api_key, shared_key)}_"


def _gen_kwargs(api_key: Optional[str], usage) -> dict:
    """Only pass extras when they are set, so plain callers see the original signature."""
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if usage is not None:
        kwargs["on_headers"] = usage.on_headers
    return kwargs


def _prepare(store, question, k, mode):
    if store is None:
        return None, "Upload a PDF first — there is nothing indexed yet."
    docs = retrieve(store, question, k or settings.default_k, mode)
    if not docs:
        return None, "No relevant passages were found in the indexed documents."
    return docs, None


def answer(store, question: str, k: int = None, system_prompt: str = None,
           mode: str = None, api_key: str = None, usage=None) -> Tuple[str, str]:
    """Return (answer_markdown, sources_markdown)."""
    docs, early = _prepare(store, question, k, mode)
    if early:
        return early, ""
    if not (api_key or settings.llm_configured):
        return NO_LLM, format_sources(docs)

    user_prompt = f"Question: {question}\n\nContext passages:\n{format_context(docs)}"
    try:
        reply = _normalise_citations(
            chat(system_prompt or DEFAULT_SYSTEM_PROMPT, user_prompt, **_gen_kwargs(api_key, usage))
        )
    except Exception as exc:
        logger.error(f"Generation failed: {type(exc).__name__}")
        reply = _error_note(exc, api_key, usage)
    return reply, format_sources(docs)


def answer_stream(store, question: str, k: int = None, system_prompt: str = None,
                  mode: str = None, api_key: str = None,
                  usage=None) -> Iterator[Tuple[str, str]]:
    """Yield (partial_answer, sources) as tokens arrive, so the UI fills in live.

    `api_key` is a visitor's own key (explicit, per call); without it the shared server
    key is used. `usage` receives rate-limit headers and 429 events."""
    docs, early = _prepare(store, question, k, mode)
    if early:
        yield early, ""
        return
    sources = format_sources(docs)
    if not (api_key or settings.llm_configured):
        yield NO_LLM, sources
        return

    user_prompt = f"Question: {question}\n\nContext passages:\n{format_context(docs)}"
    text = ""
    try:
        for delta in chat_stream(
            system_prompt or DEFAULT_SYSTEM_PROMPT, user_prompt, **_gen_kwargs(api_key, usage)
        ):
            text += delta
            yield _normalise_citations(text), sources
    except Exception as exc:
        logger.error(f"Generation failed: {type(exc).__name__}")
        note = _error_note(exc, api_key, usage)
        yield (f"{_normalise_citations(text)}\n\n{note}" if text else note), sources
