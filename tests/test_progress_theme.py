"""Loading-bar progress and the starry theme. No network, no model downloads."""

from types import SimpleNamespace

import numpy as np
import pytest

from src import models, theme


# --- embedding progress hook ---------------------------------------------------------

def _embedder():
    emb = object.__new__(models.FastEmbedEmbeddings)  # skip loading the ONNX model
    emb._model = SimpleNamespace(embed=lambda texts, **kwargs: (np.zeros(3) for _ in texts))
    return emb


def test_embedding_reports_real_counts_in_ticks():
    ticks = []
    token = models.set_embed_progress(ticks.append)
    try:
        out = _embedder().embed_documents([f"t{i}" for i in range(40)])
    finally:
        models.reset_embed_progress(token)
    assert len(out) == 40 and sum(ticks) == 40 and max(ticks) <= 16


def test_embedding_without_a_hook_is_unchanged():
    assert len(_embedder().embed_documents(["a", "b", "c"])) == 3


def test_hook_is_cleared_after_reset():
    token = models.set_embed_progress(lambda n: None)
    models.reset_embed_progress(token)
    assert models._embed_progress.get() is None


# --- the indexing handler reports progress -------------------------------------------

@pytest.fixture
def app_module(monkeypatch):
    import app

    return app


def test_index_reports_reading_then_embedding_then_done(app_module, monkeypatch):
    chunks = [SimpleNamespace(page_content=f"c{i}") for i in range(40)]
    monkeypatch.setattr(app_module, "load_and_chunk", lambda *a, **k: chunks)

    def fake_build(cs):
        tick = models._embed_progress.get()
        for n in (16, 16, 8):
            tick(n)
        return "STORE"

    monkeypatch.setattr(app_module, "build_vectorstore", fake_build)
    seen = []
    store, msg = app_module._index(
        ["x.pdf"], "recursive", 1000, 200, label="the sample",
        progress=lambda frac, desc=None: seen.append((frac, desc)),
    )
    fracs = [f for f, _ in seen]
    assert store == "STORE" and "Indexed **40 chunks**" in msg
    assert fracs == sorted(fracs) and fracs[0] < 0.1 and fracs[-1] == 1.0
    assert any("Reading" in d for _, d in seen)
    assert any("Embedding passages… 16 / 40" in d for _, d in seen)
    assert any("Embedding passages… 40 / 40" in d for _, d in seen)
    assert models._embed_progress.get() is None  # hook removed afterwards


def test_index_failure_still_removes_the_hook(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "load_and_chunk", lambda *a, **k: [SimpleNamespace(page_content="c")])

    def boom(cs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(app_module, "build_vectorstore", boom)
    store, msg = app_module._index(["x.pdf"], "recursive", 1000, 200)
    assert store is None and "Ingestion failed" in msg
    assert models._embed_progress.get() is None


def test_empty_pdf_message_and_no_progress_object_needed(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "load_and_chunk", lambda *a, **k: [])
    store, msg = app_module._index(["x.pdf"], "recursive", 1000, 200)  # progress=None works
    assert store is None and "scanned image" in msg


# --- theme ---------------------------------------------------------------------------

def test_star_field_is_deterministic_and_responsive():
    a, b = theme.stars(50, seed=7), theme.stars(50, seed=7)
    assert a == b and a != theme.stars(50, seed=8)
    assert a.count(",") == 49 and "vw" in a and "vh" in a and "px" not in a


def test_css_has_stars_loading_bar_and_reduced_motion_rule():
    css = theme.build_css()
    assert "twinkle" in css and "shoot" in css and ".progress-bar" in css
    assert "prefers-reduced-motion: reduce" in css and "animation: none" in css


def test_theme_builds_and_forces_dark():
    assert theme.build_theme() is not None
    assert "classList.add('dark')" in theme.FORCE_DARK_JS


def test_text_colors_have_readable_contrast_on_the_panel():
    def lum(hexcolor):
        c = [int(hexcolor[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    def ratio(a, b):
        hi, lo = sorted((lum(a), lum(b)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    panel = theme.PALETTE["panel_solid"]
    for name in ("text", "muted", "gold", "ice"):
        assert ratio(theme.PALETTE[name], panel) >= 4.5, name
