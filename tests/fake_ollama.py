"""A stand-in for urllib's `urlopen` that answers like Ollama, recording every request.

`/api/chat` streams as Ollama does: one JSON line per chunk (thinking first,
then the answer, then any tool calls), then a final `"done": true` line with
the counters.
"""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse

COUNTERS = {
    "total_duration": 4_200_000_000,
    "load_duration": 1_000_000_000,
    "prompt_eval_count": 120,
    "prompt_eval_duration": 600_000_000,
    "eval_count": 30,
    "eval_duration": 2_000_000_000,
}


class _Response:
    """A reply body, read whole (`http_json`) or a line at a time (`http_ndjson`).
    `on_line` sees each line just before the reader gets it."""

    def __init__(self, payload: bytes, on_line=None) -> None:
        self._body = io.BytesIO(payload)
        self._on_line = on_line
        self.closed = False

    def read(self) -> bytes:
        return self._body.read()

    def readline(self, limit: int = -1) -> bytes:
        line = self._body.readline(limit)
        if line and self._on_line:
            self._on_line(line)
        return line

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.closed = True


def ndjson(lines: list[dict]) -> bytes:
    return b"".join(json.dumps(line).encode() + b"\n" for line in lines)


def pieces(text: str, size: int = 4) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


class FakeOllama:
    def __init__(self, models=None, *, down: bool = False, chat_reply: str = '{"answer": "hi", "found": true}',
                 reject_think: bool = False, reject_think_in_stream: bool = False,
                 chat_error: str | None = None, thinking: str = "",
                 stream_error: str | None = None, stream_error_after: int = 0, cut_off: bool = False) -> None:
        self.models = models if models is not None else [
            {"name": "qwen3.5:9b", "size": 6_600_000_000, "details": {"family": "qwen35"}},
        ]
        self.down = down
        self.chat_reply = chat_reply
        self.reject_think = reject_think
        # The same refusal, sent as the stream's first line instead of an HTTP error.
        self.reject_think_in_stream = reject_think_in_stream
        self.chat_error = chat_error
        self.thinking = thinking
        # An error line after this many chunks, the way Ollama reports a failure mid-answer.
        self.stream_error = stream_error
        self.stream_error_after = stream_error_after
        # The connection drops before the final line.
        self.cut_off = cut_off
        self.on_line = None
        self.requests: list[tuple[str, dict | None]] = []
        self.responses: list[_Response] = []
        self.tool_calls: list[dict] | None = None

    def __call__(self, request, timeout=None):
        if self.down:
            raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        path = urllib.parse.urlsplit(request.full_url).path
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
            response = _Response(ndjson(self.chat_lines(body)), self.on_line)
            self.responses.append(response)
            return response
        raise _http_error(404, "not found")

    def chat_lines(self, body: dict) -> list[dict]:
        if self.reject_think_in_stream and "think" in body:
            return [{"error": '"qwen2" does not support thinking'}]
        model = body["model"]

        def chunk(**message) -> dict:
            return {"model": model, "message": {"role": "assistant", "content": "", **message}, "done": False}

        lines = [chunk(thinking=t) for t in pieces(self.thinking)]
        lines += [chunk(content=c) for c in pieces(self.chat_reply)]
        if self.tool_calls:
            lines.append(chunk(tool_calls=self.tool_calls))
        if self.stream_error is not None:
            lines.insert(self.stream_error_after, {"error": self.stream_error})
        if not self.cut_off:
            lines.append({"model": model, "message": {"role": "assistant", "content": ""}, "done": True,
                          "done_reason": "stop", **COUNTERS})
        return lines


def _http_error(code: int, message: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://127.0.0.1:11434", code, message, {},
                                  io.BytesIO(json.dumps({"error": message}).encode()))
