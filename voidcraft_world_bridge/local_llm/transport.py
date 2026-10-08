"""JSON over loopback HTTP, and the two ways a model server can fail.

Shared by every runtime adapter (`ollama.py`, `openai_compat.py`), so "it is not
running" and "it answered with an error" mean the same thing whichever server
is behind the URL.

── THE TRANSPORT IS INJECTED ─────────────────────────────────────────────────
`http_json` takes an `opener` (default: urllib) so tests drive every branch —
down, slow, malformed, refused — with no model server running.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable

Opener = Callable[..., object]


class RuntimeDown(Exception):
    """The server did not answer at all — not running, wrong URL."""


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


def http_json(url: str, body: dict | None, timeout: float, opener: Opener | None = None) -> dict:
    """GET (body None) or POST JSON to `url`, returning the parsed object."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json"})
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(request, timeout=timeout) as response:  # type: ignore[attr-defined]
            raw = response.read()
    except urllib.error.HTTPError as err:
        detail = ""
        try:
            detail = _error_text(json.loads(err.read() or b"{}"))
        except (ValueError, AttributeError):
            pass
        raise RuntimeRefused(err.code, detail or f"HTTP {err.code}") from err
    except (urllib.error.URLError, OSError, TimeoutError) as err:
        raise RuntimeDown(str(getattr(err, "reason", err))) from err
    try:
        parsed = json.loads(raw or b"{}")
    except ValueError as err:
        raise RuntimeRefused(200, f"unparseable reply: {err}") from err
    if not isinstance(parsed, dict):
        raise RuntimeRefused(200, "reply was not a JSON object")
    return parsed
