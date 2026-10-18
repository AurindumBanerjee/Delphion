"""The RAG loop: retrieve -> build a cited prompt -> generate (optionally streamed)."""

from typing import Iterator, List, Optional, Tuple

from langchain_core.documents import Document
from loguru import logger

from src.config import DEFAULT_SYSTEM_PROMPT, settings
from src.models import chat, chat_stream


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
    "and `LLM_MODEL`) to be set."
)


def _prepare(store, question, k, mode):
    if store is None:
        return None, "Upload a PDF first — there is nothing indexed yet."
    docs = retrieve(store, question, k or settings.default_k, mode)
    if not docs:
        return None, "No relevant passages were found in the indexed documents."
    return docs, None


def answer(store, question: str, k: int = None, system_prompt: str = None,
           mode: str = None) -> Tuple[str, str]:
    """Return (answer_markdown, sources_markdown)."""
    docs, early = _prepare(store, question, k, mode)
    if early:
        return early, ""
    if not settings.llm_configured:
        return NO_LLM, format_sources(docs)

    user_prompt = f"Question: {question}\n\nContext passages:\n{format_context(docs)}"
    try:
        reply = chat(system_prompt or DEFAULT_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        logger.error(f"Generation failed: {exc}")
        reply = f"Generation failed: {exc}"
    return reply, format_sources(docs)


def answer_stream(store, question: str, k: int = None, system_prompt: str = None,
                  mode: str = None) -> Iterator[Tuple[str, str]]:
    """Yield (partial_answer, sources) as tokens arrive, so the UI fills in live."""
    docs, early = _prepare(store, question, k, mode)
    if early:
        yield early, ""
        return
    sources = format_sources(docs)
    if not settings.llm_configured:
        yield NO_LLM, sources
        return

    user_prompt = f"Question: {question}\n\nContext passages:\n{format_context(docs)}"
    text = ""
    try:
        for delta in chat_stream(system_prompt or DEFAULT_SYSTEM_PROMPT, user_prompt):
            text += delta
            yield text, sources
    except Exception as exc:
        logger.error(f"Generation failed: {exc}")
        yield (text + f"\n\n_Generation failed: {exc}_"), sources
