"""Visitor-key behaviour: override, secrecy, removal, session isolation, refresh cache.
Fake client only, no network."""

import logging
from types import SimpleNamespace

import pytest
from loguru import logger

from src import keys, models, rag, usage
from tests.fake_llm import (
    SERVER_KEY, USER_HEADERS, USER_KEY, FakeProvider, Outcome,
    auth_error, not_found_error, rate_limit_error,
)


@pytest.fixture
def env(monkeypatch):
    fake = FakeProvider()
    fake_settings = SimpleNamespace(
        llm_api_key=SERVER_KEY, llm_base_url="https://api.example/v1", llm_model="m",
        temperature=0, llm_configured=True, default_k=2, retrieval_mode="hybrid",
    )
    clock = [1000.0]
    monkeypatch.setattr(models, "_make_client", fake.make_client)
    models._client.cache_clear()
    for mod in (models, rag, keys):
        monkeypatch.setattr(mod, "settings", fake_settings)
    monkeypatch.setattr(usage, "now", lambda: clock[0])
    usage.SHARED.reset()
    fake.settings, fake.clock = fake_settings, clock
    fake.outcomes[USER_KEY] = Outcome(headers=USER_HEADERS)
    yield fake
    models._client.cache_clear()
    usage.SHARED.reset()


def ask(index, user_key=None, own_book=None):
    """What app.ask does for one question; returns (final text, sources, own book)."""
    sink = usage.Sink(own=bool(user_key), book=own_book)
    out = list(rag.answer_stream(index, "Tower of Hanoi", api_key=user_key or None, usage=sink))
    return out[-1][0], out[-1][1], (sink.book if user_key else own_book)


def generation_keys(env):
    return [c["api_key"] for c in env.calls if c.get("stream")]


# --- the user key overrides the server key, and the server key stays hidden ----------

def test_user_key_overrides_server_key_and_server_key_is_never_shown(env, index):
    res = keys.set_key(f"  {USER_KEY}  ", None, None)
    assert res.accepted and res.key == USER_KEY  # whitespace stripped
    text, sources, book = ask(index, res.key, res.book)
    assert generation_keys(env) == [USER_KEY]
    shown = "\n".join([text, sources, res.message, usage.render_panel(True, book),
                       usage.render_panel(False, usage.SHARED.snapshot())])
    assert SERVER_KEY not in shown and "WXYZ1234" not in shown


def test_without_a_user_key_the_server_key_is_used(env, index):
    ask(index)
    assert generation_keys(env) == [SERVER_KEY]


def test_removing_the_key_reverts_to_the_shared_key(env, index):
    res = keys.set_key(USER_KEY, None, None)
    ask(index, res.key, res.book)
    gone = keys.remove_key()
    assert gone.key is None and gone.book is None and gone.message == "Using the shared demo key."
    ask(index, gone.key, gone.book)
    assert generation_keys(env) == [USER_KEY, SERVER_KEY]


def test_no_shared_key_but_a_user_key_still_answers(env, index):
    env.settings.llm_configured = False
    assert "No LLM endpoint configured" in ask(index)[0]
    assert ask(index, USER_KEY)[0] == "Answer [1]"


# --- the key never leaks ---------------------------------------------------------------

def test_user_key_never_appears_in_logs_or_ui_text(env, index, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    logged = []
    sink_id = logger.add(lambda m: logged.append(str(m)), level="DEBUG")
    try:
        ui = []
        res = keys.set_key(USER_KEY, None, None)
        ui += [res.message, usage.render_panel(True, res.book)]
        text, sources, book = ask(index, res.key, res.book)
        ui += [text, sources, usage.render_panel(True, book)]
        # an error path whose exception text contains the key must be redacted
        env.outcomes[USER_KEY] = Outcome(error=RuntimeError(f"boom {USER_KEY}"))
        text, sources, book = ask(index, res.key, book)
        ui += [text, sources]
        env.outcomes[USER_KEY] = Outcome(error=rate_limit_error("day"))
        ui.append(ask(index, res.key, book)[0])
        ui.append(keys.refresh(res.key, book)[1] or "")
        env.outcomes[USER_KEY] = Outcome(error=auth_error())
        ui.append(keys.set_key(USER_KEY, None, None).message)
    finally:
        logger.remove(sink_id)
    out = capsys.readouterr()
    everything = "\n".join(ui + logged + [caplog.text, out.out, out.err])
    assert USER_KEY not in everything and "gsk_USER_SECRET" not in everything
    assert "[redacted]" in "\n".join(ui)
    assert "••••5678" in res.message  # the mask (last 4 only) is the one allowed trace
    assert "SECRET_ABCD" not in everything


def test_server_key_is_redacted_in_generic_errors(env, index):
    env.outcomes[SERVER_KEY] = Outcome(error=RuntimeError(f"boom {SERVER_KEY}"))
    text = ask(index)[0]
    assert SERVER_KEY not in text and "[redacted]" in text


def test_mask_shows_at_most_the_last_four_characters():
    assert keys.mask(USER_KEY) == "••••5678"
    assert keys.mask("short-key") == "••••"  # short keys reveal nothing
    assert keys.mask(None) == ""


def test_user_key_is_not_cached_or_stored_globally(env, index):
    keys.set_key(USER_KEY, None, None)
    ask(index, USER_KEY)
    assert models._client().api_key == SERVER_KEY            # cached client is still the shared one
    assert env.settings.llm_api_key == SERVER_KEY            # settings were not mutated
    assert models._client_for(USER_KEY) is not models._client_for(USER_KEY)  # fresh per call
    assert models._client_for(None) is models._client()


# --- validation ---------------------------------------------------------------------

def test_accepted_key_reports_and_keeps_its_own_reading(env):
    res = keys.set_key(USER_KEY, None, None)
    assert res.accepted and res.message == "✅ Key accepted. Using your key: ••••5678"
    assert res.book.reading.limits.requests_remaining == 111
    assert env.probes()[0]["api_key"] == USER_KEY and len(env.probes()) == 1


def test_rejected_key_keeps_the_shared_key(env):
    env.outcomes["bad-key-123456"] = Outcome(error=auth_error())
    res = keys.set_key("bad-key-123456", None, None)
    assert not res.accepted and res.key is None
    assert "Key rejected" in res.message and "shared demo key" in res.message


def test_rejected_key_keeps_a_previous_good_key(env):
    env.outcomes["bad-key-123456"] = Outcome(error=auth_error())
    good = keys.set_key(USER_KEY, None, None)
    res = keys.set_key("bad-key-123456", good.key, good.book)
    assert res.key == USER_KEY and "Still using your key: ••••5678" in res.message


def test_key_without_model_access_is_not_accepted(env):
    env.outcomes["no-model-key-1"] = Outcome(error=not_found_error())
    res = keys.set_key("no-model-key-1", None, None)
    assert not res.accepted and "not the model" in res.message


def test_rate_limited_key_is_accepted_with_a_notice(env):
    env.outcomes[USER_KEY] = Outcome(error=rate_limit_error("minute", "9"))
    res = keys.set_key(USER_KEY, None, None)
    assert res.accepted and res.book.notice.kind == "minute"
    assert "Per-minute limit reached, resets in 9 s" in usage.render_panel(True, res.book)


def test_connection_problem_does_not_set_the_key(env):
    env.outcomes["flaky-key-0001"] = Outcome(error=ConnectionError("down"))
    res = keys.set_key("flaky-key-0001", None, None)
    assert not res.accepted and res.key is None and "Not set" in res.message


def test_empty_key_is_ignored(env):
    assert not keys.set_key("   ", None, None).accepted and env.probes() == []


# --- sessions do not leak into each other --------------------------------------------

def test_own_reading_stays_in_its_session_and_shared_reading_is_global(env, index):
    a = keys.set_key(USER_KEY, None, None)
    _, _, a_book = ask(index, a.key, a.book)

    # B is a different visitor on the shared key: A's numbers never reach the shared record
    assert usage.SHARED.snapshot().reading is None
    b_panel = usage.render_panel(False, usage.SHARED.snapshot())
    assert "111" not in b_panel and "No reading yet" in b_panel

    ask(index)  # B asks with the shared key -> the global record updates
    shared_panel = usage.render_panel(False, usage.SHARED.snapshot())
    assert "14,370" in shared_panel
    # ... and a third visitor sees the same shared reading, while A still sees only A's
    assert "14,370" in usage.render_panel(False, usage.SHARED.snapshot())
    a_panel = usage.render_panel(True, a_book)
    assert "111" in a_panel and "14,370" not in a_panel


def test_visitor_429_does_not_mark_the_shared_key_limited(env, index):
    env.outcomes[USER_KEY] = Outcome(error=rate_limit_error("day"))
    ask(index, USER_KEY, usage.UsageBook())
    assert usage.SHARED.snapshot().notice is None


# --- refresh and its cache -----------------------------------------------------------

def test_shared_refresh_is_cached_for_30_seconds(env):
    book, note = keys.refresh(None, None)
    assert note is None and len(env.probes()) == 1
    assert usage.SHARED.snapshot().reading.limits.tokens_remaining == 17997

    env.clock[0] += 10
    _, note = keys.refresh(None, None)
    assert len(env.probes()) == 1 and "once every 30 s" in note

    env.clock[0] += 21  # 31 s after the first probe
    keys.refresh(None, None)
    assert len(env.probes()) == 2


def test_a_recent_answer_counts_as_a_fresh_reading(env, index):
    ask(index)                       # real answer updates the shared record
    env.clock[0] += 5
    keys.refresh(None, None)
    assert env.probes() == []        # no extra request spent


def test_failed_shared_refresh_still_respects_the_cache(env):
    env.outcomes[SERVER_KEY] = Outcome(error=ConnectionError("down"))
    keys.refresh(None, None)
    keys.refresh(None, None)
    assert len(env.probes()) == 1


def test_shared_refresh_429_shows_the_limit_and_reset(env):
    env.outcomes[SERVER_KEY] = Outcome(error=rate_limit_error("day", "300"))
    keys.refresh(None, None)
    text = usage.render_panel(False, usage.SHARED.snapshot())
    assert "Daily limit reached, resets in 5 min 00 s" in text


def test_own_refresh_reads_the_own_key_not_the_shared_record(env):
    book, note = keys.refresh(USER_KEY, None)
    assert env.probes()[0]["api_key"] == USER_KEY
    assert book.reading.limits.requests_remaining == 111
    assert usage.SHARED.snapshot().reading is None


def test_refresh_with_no_shared_key_says_so_and_spends_nothing(env):
    env.settings.llm_configured = False
    _, note = keys.refresh(None, None)
    assert "No shared key" in note and env.probes() == []


def test_notice_expires_after_its_reset_time(env):
    book = usage.UsageBook().with_notice("minute", 20)
    assert "Per-minute limit reached, resets in 20 s" in usage.render_panel(False, book)
    env.clock[0] += 25
    assert "limit reached" not in usage.render_panel(False, book)
