"""A fake OpenAI-compatible provider: no network. Records every client built (by key) and
every request, and replays scripted headers, tokens and errors per API key."""

from types import SimpleNamespace

import httpx
import openai

SERVER_KEY = "gsk_SERVER_SECRET_WXYZ1234"
USER_KEY = "gsk_USER_SECRET_ABCD5678"
BODY_SENTINEL = "PROVIDER-BODY-SENTINEL"

SERVER_HEADERS = {
    "x-ratelimit-limit-requests": "14400",
    "x-ratelimit-remaining-requests": "14370",
    "x-ratelimit-reset-requests": "2m59.56s",
    "x-ratelimit-limit-tokens": "18000",
    "x-ratelimit-remaining-tokens": "17997",
    "x-ratelimit-reset-tokens": "7.66s",
}
USER_HEADERS = {
    "x-ratelimit-limit-requests": "500",
    "x-ratelimit-remaining-requests": "111",
    "x-ratelimit-limit-tokens": "6000",
    "x-ratelimit-remaining-tokens": "5555",
    "x-ratelimit-reset-requests": "1h2m",
    "x-ratelimit-reset-tokens": "3.5s",
}

_REQ = httpx.Request("POST", "https://api.example/v1/chat/completions")


def rate_limit_error(kind="day", retry_after="12"):
    text = {
        "day": "Rate limit reached on tokens per day (TPD): Limit 200000, Used 199990",
        "minute": "Rate limit reached on tokens per minute (TPM): Limit 6000, Used 5990",
    }[kind] + f" {BODY_SENTINEL}"
    headers = {"retry-after": retry_after} if retry_after else {}
    resp = httpx.Response(429, headers=headers, request=_REQ)
    return openai.RateLimitError(text, response=resp, body={"error": {"message": text}})


def auth_error():
    resp = httpx.Response(401, request=_REQ)
    return openai.AuthenticationError(f"Invalid API Key {BODY_SENTINEL}", response=resp, body=None)


def not_found_error():
    resp = httpx.Response(404, request=_REQ)
    return openai.NotFoundError(f"model does not exist {BODY_SENTINEL}", response=resp, body=None)


class Outcome:
    def __init__(self, headers=None, tokens=("Answer ", "[1]"), error=None):
        self.headers = SERVER_HEADERS if headers is None else headers
        self.tokens = tokens
        self.error = error


def _event(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class _Completions:
    def __init__(self, provider, api_key):
        self.provider, self.api_key = provider, api_key
        self.with_raw_response = self  # client.chat.completions.with_raw_response.create(...)

    def create(self, **kwargs):
        self.provider.calls.append({"api_key": self.api_key, **kwargs})
        out = self.provider.outcomes.get(self.api_key, self.provider.default)
        if isinstance(out, list):  # scripted sequence: use the next, repeat the last
            out = out.pop(0) if len(out) > 1 else out[0]
        if out.error is not None:
            raise out.error
        if kwargs.get("stream"):
            payload = iter([_event(t) for t in out.tokens])
        else:
            msg = SimpleNamespace(content="".join(out.tokens))
            payload = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        return SimpleNamespace(headers=out.headers, parse=lambda: payload)


class FakeClient:
    def __init__(self, provider, api_key):
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=_Completions(provider, api_key))


class FakeProvider:
    def __init__(self):
        self.clients = []   # api_key of every client built
        self.calls = []     # every request, with the key that made it
        self.outcomes = {}  # api_key -> Outcome or [Outcome, ...]
        self.default = Outcome()

    def make_client(self, api_key):
        self.clients.append(api_key)
        return FakeClient(self, api_key)

    def probes(self):
        return [c for c in self.calls if c.get("max_tokens") == 1]
