"""Offline fakes so the suite runs in CI without downloading any model."""

import hashlib
import math
import re
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class HashEmbeddings(Embeddings):
    """Bag-of-words hashed into 256 dims. Deterministic, crude, good enough to
    give shared-vocabulary texts a high cosine similarity."""

    dim = 256

    def _vec(self, text):
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class OverlapReranker:
    """Scores a passage by how many query words it contains (stand-in cross-encoder)."""

    def rerank(self, query, documents):
        q = set(re.findall(r"[a-z]+", query.lower()))
        return [len(q & set(re.findall(r"[a-z]+", d.lower()))) for d in documents]


@pytest.fixture
def docs():
    texts = [
        ("Predictive coding sends prediction errors up the cortical hierarchy.", 4),
        ("Feedback connections carry predictions of lower-level activity.", 5),
        ("The Tower of Hanoi is a simple puzzle that language models fail.", 8),
        ("Culture is the conceptual treasure communities accumulate through history.", 232),
        ("Shooting and tearing in the hip-joint, worst in the evening in bed.", 633),
        ("Shooting and tearing in the hip-joint, worst in the evening in bed.", 633),  # duplicate
    ]
    return [Document(page_content=t, metadata={"source": "x.pdf", "page": p}) for t, p in texts]


@pytest.fixture
def index(docs):
    from src.retrieval import HybridIndex

    return HybridIndex.build(docs, embeddings=HashEmbeddings(), reranker=OverlapReranker())
