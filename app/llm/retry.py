"""Retry the start of a streamed LLM call on temporary failures."""

import logging
import time
from collections.abc import Callable, Iterator
from typing import TypeVar

import httpx

from app.llm.constants import LLM_RETRY_ATTEMPTS, LLM_RETRY_BASE_DELAY

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_temporary(error: Exception) -> bool:
    """Rate limits, server errors and dropped connections are worth retrying."""
    if isinstance(error, (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)):
        return True
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


def retry_stream(name: str, start: Callable[[], Iterator[T]]) -> Iterator[T]:
    """Yield from `start()`, retrying while nothing has been yielded yet.

    Once the first piece has gone out it is too late to retry without
    repeating text, so later errors are raised as they are.
    """
    for attempt in range(LLM_RETRY_ATTEMPTS + 1):
        iterator = start()
        try:
            first = next(iterator)
        except StopIteration:
            return
        except Exception as e:
            if attempt >= LLM_RETRY_ATTEMPTS or not is_temporary(e):
                raise
            delay = LLM_RETRY_BASE_DELAY * (2**attempt)
            logger.warning("%s failed (%s), retrying in %.1fs", name, e, delay)
            time.sleep(delay)
            continue
        yield first
        yield from iterator
        return
