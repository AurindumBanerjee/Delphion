"""Provider rate-limit readings: parse headers, keep the shared-key record, format the panel.

What the numbers mean (Groq's documented headers; other providers may send none):
  x-ratelimit-*-requests  -> requests per DAY
  x-ratelimit-*-tokens    -> tokens per MINUTE
  retry-after             -> seconds, only present on a 429
The daily TOKEN limit is not in any header, so it is never shown as a balance.
"""

import re
import threading
import time
from dataclasses import dataclass, replace
from typing import Mapping, Optional

# A shared-key refresh within this many seconds of the last reading reuses the reading.
REFRESH_CACHE_SECONDS = 30

now = time.time  # indirection so tests can control the clock

_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def parse_duration(text) -> Optional[float]:
    """'2m59.56s' -> 179.56, '7.66s' -> 7.66, '12' -> 12.0; anything else -> None."""
    if text is None:
        return None
    raw = str(text).strip().lower()
    if not raw:
        return None
    parts = _DURATION.findall(raw)
    if parts and "".join(n + u for n, u in parts) == raw:
        return sum(float(n) * _UNIT_SECONDS[u] for n, u in parts)
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _int(value) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Limits:
    requests_limit: Optional[int] = None       # per day
    requests_remaining: Optional[int] = None   # per day
    requests_reset_s: Optional[float] = None
    tokens_limit: Optional[int] = None         # per minute
    tokens_remaining: Optional[int] = None     # per minute
    tokens_reset_s: Optional[float] = None

    @property
    def available(self) -> bool:
        return any(v is not None for v in self.__dict__.values())


def parse_limits(headers: Optional[Mapping[str, str]]) -> Limits:
    """Read whatever rate-limit headers are present; missing ones stay None."""
    if not headers:
        return Limits()
    h = {str(k).lower(): v for k, v in dict(headers).items()}
    return Limits(
        requests_limit=_int(h.get("x-ratelimit-limit-requests")),
        requests_remaining=_int(h.get("x-ratelimit-remaining-requests")),
        requests_reset_s=parse_duration(h.get("x-ratelimit-reset-requests")),
        tokens_limit=_int(h.get("x-ratelimit-limit-tokens")),
        tokens_remaining=_int(h.get("x-ratelimit-remaining-tokens")),
        tokens_reset_s=parse_duration(h.get("x-ratelimit-reset-tokens")),
    )


def retry_after_seconds(headers: Optional[Mapping[str, str]]) -> Optional[float]:
    if not headers:
        return None
    h = {str(k).lower(): v for k, v in dict(headers).items()}
    return parse_duration(h.get("retry-after"))


def limit_kind(exc) -> Optional[str]:
    """'day' | 'minute' | None, read from a 429's body. Used only to pick a label; the
    provider's text itself is never shown."""
    body = getattr(exc, "body", None)
    text = ""
    if isinstance(body, dict):
        err = body.get("error", body)
        text = str(err.get("message", "")) if isinstance(err, dict) else str(err)
    text = f"{text} {getattr(exc, 'message', '')}".lower()
    if "per day" in text or "(tpd)" in text or "(rpd)" in text:
        return "day"
    if "per minute" in text or "(tpm)" in text or "(rpm)" in text:
        return "minute"
    return None


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    if seconds < 1:
        return "under 1 s"
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


@dataclass(frozen=True)
class Reading:
    limits: Limits
    taken_at: float


@dataclass(frozen=True)
class Notice:
    """A 429 seen recently: which limit, and when it should clear."""
    kind: Optional[str]
    retry_after_s: Optional[float]
    taken_at: float


@dataclass(frozen=True)
class UsageBook:
    reading: Optional[Reading] = None
    notice: Optional[Notice] = None
    last_probe_at: float = 0.0

    def with_headers(self, headers) -> "UsageBook":
        return replace(self, reading=Reading(parse_limits(headers), now()), notice=None)

    def with_notice(self, kind, retry_after_s) -> "UsageBook":
        return replace(self, notice=Notice(kind, retry_after_s, now()))

    def with_probe(self) -> "UsageBook":
        return replace(self, last_probe_at=now())

    def age(self) -> Optional[float]:
        return None if self.reading is None else max(0.0, now() - self.reading.taken_at)

    def refresh_allowed(self) -> bool:
        """False if a reading or a probe happened less than REFRESH_CACHE_SECONDS ago."""
        newest = max(self.reading.taken_at if self.reading else 0.0, self.last_probe_at)
        return now() - newest >= REFRESH_CACHE_SECONDS


class SharedUsage:
    """Process-wide record for the shared server key. One instance, shared by every
    visitor, because that quota is global. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._book = UsageBook()

    def snapshot(self) -> UsageBook:
        with self._lock:
            return self._book

    def update(self, fn) -> UsageBook:
        with self._lock:
            self._book = fn(self._book)
            return self._book

    def claim_refresh(self) -> bool:
        """Atomically allow one refresh per cache window across all visitors."""
        with self._lock:
            if not self._book.refresh_allowed():
                return False
            self._book = self._book.with_probe()
            return True

    def reset(self):
        with self._lock:
            self._book = UsageBook()


SHARED = SharedUsage()


class Sink:
    """Receives rate-limit events for one request. Shared-key events go to the global
    record; own-key events go to `self.book`, which the caller keeps in session state."""

    def __init__(self, own: bool, book: Optional[UsageBook] = None):
        self.own = own
        self.book = book or UsageBook()

    def on_headers(self, headers):
        if self.own:
            self.book = self.book.with_headers(headers)
        else:
            SHARED.update(lambda b: b.with_headers(headers))

    def on_limit(self, kind, retry_after_s):
        if self.own:
            self.book = self.book.with_notice(kind, retry_after_s)
        else:
            SHARED.update(lambda b: b.with_notice(kind, retry_after_s))


def _n(value) -> str:
    return f"{value:,}"


def render_panel(own: bool, book: UsageBook) -> str:
    """Markdown for the Usage panel. Says which key it describes and what each number is."""
    t = now()
    if own:
        head = "**Usage — your own key** (this session only)"
    else:
        head = "**Usage — shared demo key** (one quota shared by all visitors)"
    lines = [head]

    n = book.notice
    if n is not None:
        left = None if n.retry_after_s is None else n.retry_after_s - (t - n.taken_at)
        if left is None or left > 0:
            label = {"day": "Daily", "minute": "Per-minute"}.get(n.kind, "Rate")
            tail = f", resets in {fmt_duration(left)}" if left is not None else ", try again later"
            lines.append(f"- ⚠️ {label} limit reached{tail}")

    r = book.reading
    if r is None:
        if n is None:
            lines.append("- No reading yet. Ask a question, or press Refresh (uses one request).")
        return "\n".join(lines)

    lim = r.limits
    if not lim.available:
        lines.append("- Limit info not available for this provider.")
    else:
        def reset(sec):
            if sec is None:
                return ""
            left = sec - (t - r.taken_at)
            return f" (resets in {fmt_duration(left)})" if left > 0 else " (reset)"

        if lim.requests_remaining is not None or lim.requests_limit is not None:
            rem = "?" if lim.requests_remaining is None else _n(lim.requests_remaining)
            tot = "" if lim.requests_limit is None else f" of {_n(lim.requests_limit)}"
            lines.append(f"- Requests left today: {rem}{tot}{reset(lim.requests_reset_s)}")
        if lim.tokens_remaining is not None or lim.tokens_limit is not None:
            rem = "?" if lim.tokens_remaining is None else _n(lim.tokens_remaining)
            tot = "" if lim.tokens_limit is None else f" of {_n(lim.tokens_limit)}"
            lines.append(f"- Tokens left this minute: {rem}{tot}{reset(lim.tokens_reset_s)}")
        lines.append("- _The daily token limit is not reported by the provider, so it is not shown._")
    lines.append(f"_Last reading {fmt_duration(t - r.taken_at) or 'just now'} ago._")
    return "\n".join(lines)
