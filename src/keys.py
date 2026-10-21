"""Visitor API-key handling, kept free of UI code so it can be tested without Gradio.

A visitor's key lives only in the caller's session state. Nothing here writes it to
disk, to the environment, to a cache or to a log, and no message built here contains
it (the status line shows at most the last 4 characters).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from loguru import logger
from openai import (
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from src import models, usage
from src.config import settings
from src.usage import SHARED, UsageBook


def mask(key: Optional[str]) -> str:
    """'••••abcd'. Keys shorter than 12 characters reveal nothing at all."""
    if not key:
        return ""
    return "••••" + key[-4:] if len(key) >= 12 else "••••"


def in_use(key: Optional[str]) -> str:
    if key:
        return f"Using your key: {mask(key)}"
    return "Using the shared demo key."


def still(key: Optional[str]) -> str:
    if key:
        return f"Still using your key: {mask(key)}."
    return "Still using the shared demo key."


@dataclass(frozen=True)
class KeyResult:
    key: Optional[str]            # key to hold in session state afterwards
    book: Optional[UsageBook]     # the visitor's own usage record
    message: str                  # status line for the UI
    accepted: bool = False


def _limit_info(exc) -> Tuple[Optional[str], Optional[float]]:
    retry = usage.retry_after_seconds(getattr(getattr(exc, "response", None), "headers", None))
    return usage.limit_kind(exc), retry


def set_key(raw: Optional[str], current_key: Optional[str],
            current_book: Optional[UsageBook]) -> KeyResult:
    """Validate a pasted key with one cheap call. On rejection the previous state is kept."""
    key = (raw or "").strip()
    if not key:
        return KeyResult(current_key, current_book, f"Paste a key first. {in_use(current_key)}")
    try:
        headers = models.probe(api_key=key)
    except (AuthenticationError, PermissionDeniedError):
        return KeyResult(current_key, current_book, f"❌ Key rejected. {still(current_key)}")
    except NotFoundError:
        return KeyResult(
            current_key, current_book,
            f"❌ The provider accepted the key but not the model this demo uses. {still(current_key)}",
        )
    except RateLimitError as exc:
        kind, retry = _limit_info(exc)
        book = UsageBook().with_notice(kind, retry)
        return KeyResult(key, book, f"✅ Key accepted, but it is rate-limited right now. {in_use(key)}", True)
    except Exception as exc:
        logger.warning(f"Key check failed: {type(exc).__name__}")
        return KeyResult(
            current_key, current_book,
            f"Could not check the key right now (connection problem). Not set. {still(current_key)}",
        )
    return KeyResult(key, UsageBook().with_headers(headers), f"✅ Key accepted. {in_use(key)}", True)


def remove_key() -> KeyResult:
    return KeyResult(None, None, in_use(None))


def refresh(user_key: Optional[str], book: Optional[UsageBook]) -> Tuple[Optional[UsageBook], Optional[str]]:
    """Read the limits with one max_tokens=1 request. Returns (own book, note).

    For the shared key the reading is global, so a refresh inside the cache window
    reuses the existing reading instead of spending another request."""
    if user_key:
        book = book or UsageBook()
        try:
            return book.with_headers(models.probe(api_key=user_key)).with_probe(), None
        except RateLimitError as exc:
            return book.with_notice(*_limit_info(exc)), None
        except (AuthenticationError, PermissionDeniedError):
            return book, "Your key was rejected by the provider."
        except Exception as exc:
            logger.warning(f"Usage refresh failed: {type(exc).__name__}")
            return book, "Could not read usage right now."

    if not settings.llm_configured:
        return book, "No shared key is configured, so there is no usage to show."
    if not SHARED.claim_refresh():
        return book, (
            f"Showing the latest reading: the shared key is refreshed at most once every "
            f"{usage.REFRESH_CACHE_SECONDS} s."
        )
    try:
        headers = models.probe()
        SHARED.update(lambda b: b.with_headers(headers))
    except RateLimitError as exc:
        kind, retry = _limit_info(exc)
        SHARED.update(lambda b: b.with_notice(kind, retry))
    except (AuthenticationError, PermissionDeniedError):
        return book, "The shared demo key was rejected by the provider."
    except Exception as exc:
        logger.warning(f"Usage refresh failed: {type(exc).__name__}")
        return book, "Could not read usage right now."
    return book, None
