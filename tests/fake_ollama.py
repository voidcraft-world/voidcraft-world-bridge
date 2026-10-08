"""A stand-in for urllib's `urlopen` that answers like Ollama, recording every request."""
from __future__ import annotations

import io
import json
import urllib.error


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


class FakeOllama:
    def __init__(self, models=None, *, down: bool = False, chat_reply: str = '{"answer": "hi", "found": true}',
                 reject_think: bool = False, chat_error: str | None = None) -> None:
        self.models = models if models is not None else [
            {"name": "qwen3.5:9b", "size": 6_600_000_000, "details": {"family": "qwen35"}},
        ]
        self.down = down
        self.chat_reply = chat_reply
        self.reject_think = reject_think
        self.chat_error = chat_error
        self.requests: list[tuple[str, dict | None]] = []
        self.tool_calls: list[dict] | None = None

    def __call__(self, request, timeout=None):
        if self.down:
            raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        path = request.full_url.split("11434", 1)[-1] if "11434" in request.full_url else request.full_url
        body = json.loads(request.data) if request.data else None
        self.requests.append((path, body))
        if path == "/api/tags":
            return _Response(json.dumps({"models": self.models}).encode())
        if path == "/api/version":
            return _Response(b'{"version": "0.12.3"}')
        if path == "/api/chat":
            if self.reject_think and body and "think" in body:
                raise _http_error(400, '"qwen2" does not support thinking')
            if self.chat_error:
                raise _http_error(500, self.chat_error)
            return _Response(json.dumps({
                "model": body["model"],
                "message": {"role": "assistant", "content": self.chat_reply,
                            **({"tool_calls": self.tool_calls} if self.tool_calls else {})},
                "done": True,
                "total_duration": 4_200_000_000,
                "load_duration": 1_000_000_000,
                "prompt_eval_count": 120,
                "eval_count": 30,
            }).encode())
        raise _http_error(404, "not found")


def _http_error(code: int, message: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://127.0.0.1:11434", code, message, {},
                                  io.BytesIO(json.dumps({"error": message}).encode()))
