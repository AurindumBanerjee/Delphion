"""Retrieval strategies: dense, BM25, hybrid (RRF), and hybrid + cross-encoder rerank.

Why four modes instead of one:
  * dense  — embedding similarity. Good at paraphrase, weak on rare exact terms
             (names, numbers, jargon) because those carry little weight in a
             384-dim sentence vector.
  * bm25   — classic lexical ranking. The mirror image: exact terms win,
             paraphrase loses.
  * hybrid — fuse both rankings with Reciprocal Rank Fusion. RRF uses ranks,
             not scores, so it needs no calibration between the two systems.
  * rerank — take the fused top-N and re-score each (query, passage) pair with
             a cross-encoder, which reads both texts together. Slower per pair,
             so it only ever sees a short candidate list.

The index object lives in Gradio session state, exactly like the old Chroma
store did, so per-session isolation is unchanged.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.documents import Document
from loguru import logger

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be but by for from has have he her his i in is it its of on or "
    "she that the their them they this to was were which with you your we our not".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]


def rrf(rankings: List[List[int]], k: int = 60) -> List[int]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank).

    k=60 is the constant from Cormack et al. (2009); it damps the influence of
    the very top ranks so one system cannot dominate on its own.
    """
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: -scores[i])


def dedupe(chunks: List[Document]) -> List[Document]:
    """Drop chunks whose normalised text is identical (repeated headers, footers,
    boilerplate pages). Duplicates waste top-k slots with the same passage."""
    seen, out = set(), []
    for chunk in chunks:
        key = hashlib.sha1(" ".join(chunk.page_content.split()).lower().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(chunk)
    if len(out) < len(chunks):
        logger.info(f"dedupe: dropped {len(chunks) - len(out)} duplicate chunks")
    return out


def _clean(metadata: dict) -> dict:
    """Chroma only accepts str/int/float/bool values; drop or coerce the rest."""
    out = {}
    for key, value in (metadata or {}).items():
        if value is None or value == "":
            continue
        out[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return out


@dataclass
class HybridIndex:
    """One index, four ways to query it."""

    docs: List[Document]
    dense: object  # langchain Chroma, or anything with similarity_search(q, k)
    bm25: object  # rank_bm25.BM25Okapi
    reranker: Optional[object] = None
    candidates: int = 20
    _pos: Dict[str, int] = field(default_factory=dict)

    # ---- construction -------------------------------------------------
    @classmethod
    def build(cls, chunks: List[Document], embeddings=None, reranker=None, candidates: int = 20):
        from langchain_chroma import Chroma
        from rank_bm25 import BM25Okapi

        if not chunks:
            raise ValueError("No chunks to index")
        if embeddings is None:
            from src.models import get_embeddings

            embeddings = get_embeddings()

        chunks = dedupe(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]
        for chunk, cid in zip(chunks, ids):
            chunk.metadata = _clean(chunk.metadata)
            chunk.metadata["chunk_id"] = cid

        dense = Chroma.from_texts(
            texts=[c.page_content for c in chunks],
            embedding=embeddings,
            metadatas=[c.metadata for c in chunks],
            ids=ids,
            collection_name=f"pdf_chunks_{uuid.uuid4().hex[:8]}",
            collection_metadata={"hnsw:space": "cosine"},
        )
        bm25 = BM25Okapi([tokenize(c.page_content) or ["_"] for c in chunks])
        index = cls(docs=chunks, dense=dense, bm25=bm25, reranker=reranker, candidates=candidates)
        index._pos = {cid: i for i, cid in enumerate(ids)}
        logger.info(f"HybridIndex: {len(chunks)} chunks (dense + BM25)")
        return index

    # ---- individual rankers -------------------------------------------
    def _dense_ranking(self, query: str, n: int) -> List[int]:
        hits = self.dense.similarity_search(query, k=min(n, len(self.docs)))
        return [self._pos[h.metadata["chunk_id"]] for h in hits if h.metadata.get("chunk_id") in self._pos]

    def _bm25_ranking(self, query: str, n: int) -> List[int]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [i for i in order[:n] if scores[i] > 0]

    def _rerank(self, query: str, idxs: List[int]) -> List[int]:
        if not idxs or self.reranker is None:
            return idxs
        texts = [self.docs[i].page_content for i in idxs]
        scores = list(self.reranker.rerank(query, texts))
        return [i for _, i in sorted(zip(scores, idxs), key=lambda p: -p[0])]

    # ---- public API ---------------------------------------------------
    def search(self, query: str, k: int = 4, mode: str = "hybrid_rerank") -> List[Document]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        n = max(k, self.candidates)

        if mode == "dense":
            order = self._dense_ranking(query, k)
        elif mode == "bm25":
            order = self._bm25_ranking(query, k)
        else:
            order = rrf([self._dense_ranking(query, n), self._bm25_ranking(query, n)])[:n]
            if mode == "hybrid_rerank":
                if self.reranker is None:
                    logger.warning("hybrid_rerank requested but no reranker loaded; using hybrid")
                order = self._rerank(query, order)

        return [self.docs[i] for i in order[:k]]

    # Keeps the old call sites (and the eval harness) working unchanged.
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return self.search(query, k, mode="dense")
