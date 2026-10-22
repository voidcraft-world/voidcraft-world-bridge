"""JSON over loopback HTTP, and the two ways a model server can fail.

Shared by every runtime adapter (`ollama.py`, `openai_compat.py`), so "it is not
running" and "it answered with an error" mean the same thing whichever server
is behind the URL. `http_json` reads one reply whole; `http_ndjson` reads a reply
streamed as one JSON object per line, each line as it arrives.

── THE TRANSPORT IS INJECTED ─────────────────────────────────────────────────
Both take an `opener` (default: urllib) so tests drive every branch — down,
slow, malformed, refused, cut off mid-stream — with no model server running.

── A STREAM KEEPS TWO LIMITS ─────────────────────────────────────────────────
A one-shot reply arrives in one piece, so its socket timeout bounds the whole
call. A stream does not: a server can keep sending a token every few seconds
for an hour. So `http_ndjson`'s `timeout` is two limits at once: the socket's
per-read timeout (a stalled connection fails instead of hanging forever) and a
deadline checked as each line arrives (a reply that keeps trickling ends there).
The per-read timeout cannot be shorter than the deadline, because the longest
honest silence is the FIRST one — the server paging the model in and reading
the prompt before it writes anything — and a socket's timeout is fixed once it
opens. So a server that stalls just before the deadline is caught one read
timeout later: at worst twice `timeout`, never forever.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

Opener = Callable[..., Any]
"""Anything that behaves like `urllib.request.urlopen`: called with a Request and
a timeout, returns a context manager whose value has `.read()` and `.readline()`."""

MAX_LINE_BYTES = 1024 * 1024
"""One streamed line. Ollama's lines are a token each, a few hundred bytes; the
longest is a tool call with its arguments. A line past this is a server
misbehaving, not a reply."""


class RuntimeDown(Exception):
    """The server did not answer at all — not running, wrong URL — or stopped mid-reply."""


class RuntimeRefused(Exception):
    """The server answered with an error. `.message` is its own text."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _error_text(payload: object) -> str:
    """The message out of an error body: Ollama sends `{"error": "…"}`, the
    OpenAI shape `{"error": {"message": "…"}}`."""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, dict):
        error = error.get("message")
    return error if isinstance(error, str) else ""


def _reason(err: BaseException) -> str:
    return str(getattr(err, "reason", err))


def _open(url: str, body: dict | None, timeout: float, opener: Opener | None) -> Any:
    """GET (body None) or POST JSON to `url`; the open response, or one of the two failures."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json"})
    open_fn = opener or urllib.request.urlopen
    try:
        return open_fn(request, timeout=timeout)
    except urllib.error.HTTPError as err:
        detail = ""
        with contextlib.suppress(ValueError, AttributeError):
            detail = _error_text(json.loads(err.read() or b"{}"))
        raise RuntimeRefused(err.code, detail or f"HTTP {err.code}") from err
    except (urllib.error.URLError, OSError, TimeoutError) as err:
        raise RuntimeDown(_reason(err)) from err


def _object(raw: bytes, what: str) -> dict:
    try:
        parsed = json.loads(raw or b"{}")
    except ValueError as err:
        raise RuntimeRefused(200, f"unparsable {what}: {err}") from err
    if not isinstance(parsed, dict):
        raise RuntimeRefused(200, f"{what} was not a JSON object")
    return parsed


def http_json(url: str, body: dict | None, timeout: float, opener: Opener | None = None) -> dict:
    """GET (body None) or POST JSON to `url`, returning the parsed object."""
    try:
        with _open(url, body, timeout, opener) as response:
            raw = response.read()
    except (OSError, TimeoutError, http.client.HTTPException) as err:
        raise RuntimeDown(_reason(err)) from err
    return _object(raw, "reply")


@contextlib.contextmanager
def http_ndjson(url: str, body: dict, timeout: float, opener: Opener | None = None, *,
                clock: Callable[[], float] = time.monotonic) -> Iterator[Iterator[dict]]:
    """POST JSON to `url` and read the reply as newline-delimited JSON, a line at a time.

        with http_ndjson(url, body, timeout) as lines:
            for line in lines: …

    Opening fails like `http_json`. After that, a line that is an error body
    (`{"error": …}` — what a server sends when it fails after it began
    streaming) raises RuntimeRefused, as an HTTP error would have; a connection
    that breaks, stalls or outlives `timeout` raises RuntimeDown (the module
    note). Leaving the block closes the connection, which is also how a server
    is told to stop generating.
    """
    deadline = clock() + timeout
    with _open(url, body, timeout, opener) as response:
        yield _lines(response, deadline, timeout, clock)


def _lines(response: Any, deadline: float, timeout: float, clock: Callable[[], float]) -> Iterator[dict]:
    while True:
        try:
            raw = response.readline(MAX_LINE_BYTES + 1)
        except (OSError, TimeoutError, ValueError, http.client.HTTPException) as err:
            raise RuntimeDown(f"the reply broke off: {_reason(err)}") from err
        if not raw:
            return
        if clock() > deadline:
            raise RuntimeDown(f"no complete reply within {timeout:.0f} s")
        if len(raw) > MAX_LINE_BYTES:
            raise RuntimeRefused(200, f"a streamed line longer than {MAX_LINE_BYTES} bytes")
        if not raw.strip():
            continue
        line = _object(raw, "stream line")
        error = _error_text(line)
        if error:
            raise RuntimeRefused(200, error)
        yield line
