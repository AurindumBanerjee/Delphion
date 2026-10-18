import math

from langchain_core.documents import Document

from eval.metrics import aggregate, is_relevant, normalize, score

Q = {"question": "?", "answer_span": "Tower of\nHanoi", "source": "x.pdf", "page": 8}


def d(text, page=1):
    return Document(page_content=text, metadata={"source": "x.pdf", "page": page})


def test_normalize_collapses_pdf_line_breaks():
    assert normalize("  Tower  of\n Hanoi ") == "tower of hanoi"


def test_relevant_by_span_or_page():
    assert is_relevant(d("the TOWER of hanoi puzzle"), Q)
    assert is_relevant(d("unrelated", page=8), Q)
    assert not is_relevant(d("unrelated", page=9), Q)


def test_score_rank_two():
    s = score([d("nope"), d("tower of hanoi"), d("nope")], Q, k=3)
    assert s["hit"] == 1 and s["rank"] == 2
    assert s["mrr"] == 0.5
    assert math.isclose(s["ndcg"], 1 / math.log2(3))
    assert math.isclose(s["precision"], 1 / 3)


def test_score_respects_cutoff():
    assert score([d("nope"), d("tower of hanoi")], Q, k=1)["hit"] == 0


def test_aggregate_means():
    rows = [{"hit": 1, "mrr": 1, "precision": 1, "ndcg": 1}, {"hit": 0, "mrr": 0, "precision": 0, "ndcg": 0}]
    assert aggregate(rows)["hit"] == 0.5
    assert aggregate([])["mrr"] == 0.0
