"""Index construction for the app.

The index lives for the life of one user session and is held in Gradio state.
There is deliberately no persist_directory: the old on-disk Chroma was the
source of the "embed succeeded, retrieval empty" bug when it ran across
multiple serverless containers.
"""

from typing import List

from langchain_core.documents import Document

from src.config import settings
from src.retrieval import HybridIndex


def build_vectorstore(chunks: List[Document], with_reranker: bool = True) -> HybridIndex:
    """Embed chunks, build the BM25 side, attach the reranker; return the index."""
    reranker = None
    if with_reranker and settings.retrieval_mode == "hybrid_rerank":
        from src.models import get_reranker

        reranker = get_reranker()
    return HybridIndex.build(chunks, reranker=reranker, candidates=settings.rerank_candidates)
