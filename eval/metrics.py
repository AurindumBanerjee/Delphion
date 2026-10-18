"""Retrieval metrics.

Ground truth is expressed as a short verbatim span from the source document,
not as a chunk id. That choice is deliberate: chunk ids change every time the
chunking strategy or size changes, which would make the question set useless
for exactly the comparison it exists to support. A span survives re-chunking.
"""

import math
import re
from typing import Dict, List


def normalize(text: str) -> str:
    """Collapse whitespace and case so PDF line breaks don't defeat matching."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def is_relevant(doc, question: Dict) -> bool:
    """A retrieved chunk is relevant if it satisfies any criterion the question sets."""
    span = question.get("answer_span")
    if span and normalize(span) in normalize(doc.page_content):
        return True

    source, page = question.get("source"), question.get("page")
    if source and page is not None:
        if doc.metadata.get("source") == source and doc.metadata.get("page") == page:
            return True

    return False


def score(docs: List, question: Dict, k: int) -> Dict[str, float]:
    """Metrics for one question at cutoff k.

    hit@k        did any relevant chunk make the cut at all
    mrr          1/rank of the first relevant chunk (0 if none) — rewards ranking it first
    precision@k  what fraction of what we retrieved was relevant (signal-to-noise)
    ndcg@k       discounted gain, normalised against the ideal of one relevant chunk at rank 1
    """
    top = docs[:k]
    flags = [is_relevant(d, question) for d in top]

    first = next((i + 1 for i, hit in enumerate(flags) if hit), None)

    return {
        "hit": 1.0 if first else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "precision": sum(flags) / k if k else 0.0,
        "ndcg": 1.0 / math.log2(first + 1) if first else 0.0,
        "rank": first or 0,
    }


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {key: 0.0 for key in ("hit", "mrr", "precision", "ndcg")}
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in ("hit", "mrr", "precision", "ndcg")
    }
