from src import rag


def test_no_store_asks_for_upload():
    reply, sources = rag.answer(None, "anything")
    assert "Upload a PDF" in reply and sources == ""


def test_context_is_numbered_with_citations(index):
    docs = index.search("Tower of Hanoi", 2, "hybrid")
    ctx = rag.format_context(docs)
    assert ctx.startswith("[1] (x.pdf, p.") and "[2]" in ctx


def test_without_llm_key_retrieval_still_shown(index, monkeypatch):
    monkeypatch.setattr(rag, "settings", type("S", (), {"llm_configured": False, "default_k": 3,
                                                          "retrieval_mode": "hybrid"})())
    reply, sources = rag.answer(index, "hip joint pain")
    assert "No LLM endpoint" in reply and "page 633" in sources


def test_stream_yields_growing_text(index, monkeypatch):
    monkeypatch.setattr(rag, "settings", type("S", (), {"llm_configured": True, "default_k": 2,
                                                          "retrieval_mode": "hybrid"})())
    monkeypatch.setattr(rag, "chat_stream", lambda s, u: iter(["The ", "answer ", "[1]"]))
    chunks = [text for text, _ in rag.answer_stream(index, "Tower of Hanoi")]
    assert chunks == ["The ", "The answer ", "The answer [1]"]


def test_stream_reports_generation_errors(index, monkeypatch):
    monkeypatch.setattr(rag, "settings", type("S", (), {"llm_configured": True, "default_k": 2,
                                                          "retrieval_mode": "dense"})())

    def boom(s, u):
        yield "partial"
        raise RuntimeError("rate limited")

    monkeypatch.setattr(rag, "chat_stream", boom)
    final = list(rag.answer_stream(index, "Hanoi"))[-1][0]
    assert final.startswith("partial") and "rate limited" in final
