"""Header parsing, panel wording and 429 mapping. Fake client only, no network."""

from types import SimpleNamespace

import pytest

from src import models, rag, usage
from tests.fake_llm import (
    BODY_SENTINEL, SERVER_HEADERS, SERVER_KEY, USER_KEY, FakeProvider, Outcome,
    auth_error, rate_limit_error,
)


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider()
    fake_settings = SimpleNamespace(
        llm_api_key=SERVER_KEY, llm_base_url="https://api.example/v1", llm_model="m",
        temperature=0, llm_configured=True, default_k=2, retrieval_mode="hybrid",
    )
    monkeypatch.setattr(models, "_make_client", fake.make_client)
    models._client.cache_clear()
    for mod in (models, rag):
        monkeypatch.setattr(mod, "settings", fake_settings)
    usage.SHARED.reset()
    yield fake
    models._client.cache_clear()
    usage.SHARED.reset()


def run(index, api_key=None, sink=None):
    return list(rag.answer_stream(index, "Tower of Hanoi", api_key=api_key, usage=sink))


# --- parsing -------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("2m59.56s", 179.56), ("7.66s", 7.66), ("1h2m", 3720.0), ("12", 12.0),
    ("250ms", 0.25), ("", None), (None, None), ("soon", None), ("-3", None),
])
def test_parse_duration(text, expected):
    got = usage.parse_duration(text)
    assert got is None if expected is None else got == pytest.approx(expected)


def test_parse_limits_all_headers_present():
    lim = usage.parse_limits(SERVER_HEADERS)
    assert (lim.requests_limit, lim.requests_remaining) == (14400, 14370)
    assert (lim.tokens_limit, lim.tokens_remaining) == (18000, 17997)
    assert lim.requests_reset_s == pytest.approx(179.56) and lim.tokens_reset_s == pytest.approx(7.66)
    assert lim.available


def test_parse_limits_is_case_insensitive():
    lim = usage.parse_limits({"X-RateLimit-Remaining-Tokens": "5"})
    assert lim.tokens_remaining == 5


def test_parse_limits_missing_headers():
    for empty in (None, {}):
        lim = usage.parse_limits(empty)
        assert not lim.available and lim.requests_remaining is None


def test_parse_limits_partial_and_garbage_headers():
    lim = usage.parse_limits({"x-ratelimit-remaining-tokens": "17997",
                              "x-ratelimit-limit-requests": "lots"})
    assert lim.tokens_remaining == 17997 and lim.requests_limit is None and lim.available


# --- panel wording -------------------------------------------------------------------

def test_panel_labels_say_what_each_number_measures():
    book = usage.UsageBook().with_headers(SERVER_HEADERS)
    text = usage.render_panel(False, book)
    assert "shared demo key" in text and "all visitors" in text
    assert "Requests left today: 14,370 of 14,400" in text
    assert "Tokens left this minute: 17,997 of 18,000" in text
    assert "daily token limit is not reported" in text
    assert "token balance" not in text.lower()


def test_panel_names_the_own_key_source():
    text = usage.render_panel(True, usage.UsageBook().with_headers(SERVER_HEADERS))
    assert "your own key" in text and "shared demo key" not in text


def test_panel_without_headers_does_not_invent_numbers():
    text = usage.render_panel(False, usage.UsageBook().with_headers({}))
    assert "Limit info not available for this provider" in text
    assert "Requests left" not in text and "Tokens left" not in text


def test_panel_with_partial_headers_shows_only_what_exists():
    text = usage.render_panel(False, usage.UsageBook().with_headers({"x-ratelimit-remaining-tokens": "9"}))
    assert "Tokens left this minute: 9" in text and "Requests left today" not in text


def test_panel_before_any_reading_mentions_cost_of_refresh():
    assert "uses one request" in usage.render_panel(False, usage.UsageBook())


def test_reset_countdown_uses_time_since_the_reading(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(usage, "now", lambda: clock[0])
    book = usage.UsageBook().with_headers({"x-ratelimit-remaining-tokens": "5", "x-ratelimit-reset-tokens": "20s"})
    assert "resets in 20 s" in usage.render_panel(False, book)
    clock[0] += 8
    assert "resets in 12 s" in usage.render_panel(False, book)
    clock[0] += 30
    assert "(reset)" in usage.render_panel(False, book)


# --- 429 mapping ---------------------------------------------------------------------

def test_shared_key_daily_limit_message(provider, index):
    provider.outcomes[SERVER_KEY] = Outcome(error=rate_limit_error("day", "12"))
    sink = usage.Sink(own=False)
    final = run(index, sink=sink)[-1][0]
    assert "Daily limit reached, resets in 12 s" in final
    assert "shared demo key" in final and "your own key's limit" not in final
    assert BODY_SENTINEL not in final and "Limit 200000" not in final
    notice = usage.SHARED.snapshot().notice
    assert notice.kind == "day" and notice.retry_after_s == 12


def test_own_key_minute_limit_says_your_key_not_the_demo(provider, index):
    provider.outcomes[USER_KEY] = Outcome(error=rate_limit_error("minute", "7"))
    sink = usage.Sink(own=True)
    final = run(index, api_key=USER_KEY, sink=sink)[-1][0]
    assert "Per-minute limit reached, resets in 7 s" in final
    assert "your own key's limit, not the demo's quota" in final
    assert BODY_SENTINEL not in final
    assert sink.book.notice.kind == "minute"
    assert usage.SHARED.snapshot().notice is None  # a visitor's 429 never marks the shared key


def test_limit_without_retry_after_says_try_again_later(provider, index):
    provider.outcomes[SERVER_KEY] = Outcome(error=rate_limit_error("day", None))
    assert "try again later" in run(index)[-1][0]


def test_rejected_keys_are_worded_for_whose_key_it_is(provider, index):
    provider.outcomes[SERVER_KEY] = Outcome(error=auth_error())
    provider.outcomes[USER_KEY] = Outcome(error=auth_error())
    shared = run(index)[-1][0]
    own = run(index, api_key=USER_KEY)[-1][0]
    assert "shared demo key was rejected" in shared and BODY_SENTINEL not in shared
    assert "Your key was rejected" in own and "shared demo key was rejected" not in own


def test_stream_records_headers_from_the_real_response(provider, index):
    sink = usage.Sink(own=False)
    out = run(index, sink=sink)
    assert out[-1][0] == "Answer [1]"
    assert usage.SHARED.snapshot().reading.limits.tokens_remaining == 17997


def test_provider_without_headers_yields_unavailable(provider, index):
    provider.outcomes[SERVER_KEY] = Outcome(headers={})
    run(index, sink=usage.Sink(own=False))
    assert "Limit info not available" in usage.render_panel(False, usage.SHARED.snapshot())
