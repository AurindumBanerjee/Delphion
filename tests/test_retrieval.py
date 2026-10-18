import pytest

from src.retrieval import MODES, dedupe, rrf, tokenize


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("The Tower of Hanoi, again!") == ["tower", "hanoi", "again"]


def test_rrf_rewards_agreement_between_rankers():
    # item 2 is 2nd in both lists; items 1 and 3 top only one list each
    fused = rrf([[1, 2, 3], [3, 2, 1]])
    assert fused[0] == 2 or set(fused[:3]) == {1, 2, 3}
    assert rrf([[5, 6], [5, 7]])[0] == 5  # first in both -> first overall


def test_rrf_handles_disjoint_lists():
    assert set(rrf([[1], [2]])) == {1, 2}


def test_dedupe_removes_whitespace_and_case_variants(docs):
    out = dedupe(docs)
    assert len(out) == len(docs) - 1


def test_index_deduplicates(index, docs):
    assert len(index.docs) == len(docs) - 1


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_finds_an_exact_term(index, mode):
    hits = index.search("Which puzzle is Tower of Hanoi?", k=2, mode=mode)
    assert hits and "Hanoi" in hits[0].page_content


def test_bm25_returns_nothing_for_unknown_words(index):
    assert index.search("zzzz qqqq", k=3, mode="bm25") == []


def test_k_is_respected(index):
    assert len(index.search("hip joint pain", k=2, mode="hybrid_rerank")) == 2


def test_rerank_without_model_falls_back_to_hybrid(docs):
    from src.retrieval import HybridIndex
    from tests.conftest import HashEmbeddings

    idx = HybridIndex.build(docs, embeddings=HashEmbeddings(), reranker=None)
    assert idx.search("hip joint", 2, "hybrid_rerank") == idx.search("hip joint", 2, "hybrid")


def test_unknown_mode_rejected(index):
    with pytest.raises(ValueError):
        index.search("x", mode="magic")


def test_similarity_search_is_dense_alias(index):
    q = "feedback predictions"
    assert index.similarity_search(q, 2) == index.search(q, 2, "dense")
