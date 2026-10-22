"""What a running chat job shows while the model writes: the `progress` of `GET /job`.

One answer from a local model takes 5–60 s, and a page that can say only
"running" for all of it looks hung. So an adapter that streams (`ollama.py`)
feeds each streamed line into a `ChatProgress`, and the job store (`jobs.py`)
publishes what it reports:

    {"phase": "reading" | "writing", "text": str, "output_tokens": int,
     "thinking": bool, "read_ms": int | None}

- `phase` is "reading" until the first streamed delta — the server paging the
  model in, then reading the prompt (prefill) — and "writing" from then on.
- `text` is the answer so far, whole (not a delta), cut at MAX_TEXT_CHARS.
- `output_tokens` counts streamed chunks. Ollama sends about one token per
  chunk, so it approximates the true count and runs a little behind it (a tool
  call arrives as ONE chunk); the finished `result.output_tokens` is the
  server's own count.
- `thinking` is true while a reasoning model is writing its thinking and none
  of the answer yet. The thinking itself is not shown, as the result never
  carried it either.
- `read_ms` is how long reading took — from the request going out to the
  first delta — and null until then.

A runtime that does not stream reports nothing, and its job's `progress` stays
null: a page treats null as "running, no detail".

── WHY THROTTLED ─────────────────────────────────────────────────────────────
A model writes 20–100 tokens a second and a page polls about every 400 ms, so
reporting every token would take the store's lock that often for changes no
page could see. A report goes out at most every REPORT_INTERVAL_SECONDS, at
once when the picture changes in kind (the first delta, the thinking giving
way to the answer), and once more at the end with whatever the throttle held.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from voidcraft_world_bridge.local_llm.jobs import Report

READING, WRITING = "reading", "writing"

REPORT_INTERVAL_SECONDS = 0.15

MAX_TEXT_CHARS = 64_000
"""A backstop only: `max_tokens` (at most 4096) bounds an answer well under it.
The finished `result.text` is never cut."""


class ChatProgress:
    """One answer as it streams in.

    It holds the whole text, which the adapter's result reads (`text`), and
    sends a capped snapshot to `report`, throttled (the module note). With no
    `report` it only accumulates, so an adapter has one code path either way.
    Lives on the job's worker thread; `report` is where it crosses to readers.
    """

    def __init__(self, report: Report | None = None, *, clock: Callable[[], float] = time.monotonic,
                 interval: float = REPORT_INTERVAL_SECONDS) -> None:
        self._report = report
        self._clock = clock
        self._interval = interval
        self._started = clock()
        self._parts: list[str] = []
        self._chunks = 0
        self._thinking = False
        self._read_ms: int | None = None
        self._sent_at: float | None = None
        self._held = False  # a change the throttle has not sent yet

    @property
    def writing(self) -> bool:
        """True once the first delta has arrived."""
        return self._read_ms is not None

    @property
    def text(self) -> str:
        if len(self._parts) > 1:
            self._parts = ["".join(self._parts)]  # joined once per read, not once per token
        return self._parts[0] if self._parts else ""

    def begin(self) -> None:
        """Report "reading" now, before the request goes out: paging the model in
        and reading the prompt are the slow part, and the page should say so."""
        self._send(self._clock())

    def add(self, content: str = "", thinking: str = "", *, tool_calls: bool = False) -> None:
        """One streamed line's delta. A line with none of the three (Ollama's
        final line) is not a chunk and changes nothing."""
        if not (content or thinking or tool_calls):
            return
        now = self._clock()
        first = self._read_ms is None
        if first:
            self._read_ms = int((now - self._started) * 1000)
        self._chunks += 1
        if content:
            self._parts.append(content)
        was_thinking = self._thinking
        self._thinking = (was_thinking or bool(thinking)) and not self._parts
        if first or self._thinking != was_thinking or self._sent_at is None \
                or now - self._sent_at >= self._interval:
            self._send(now)
        else:
            self._held = True

    def finish(self) -> None:
        """The stream ended: send what the throttle held back."""
        if self._held:
            self._send(self._clock())

    def snapshot(self) -> dict:
        return {
            "phase": WRITING if self.writing else READING,
            "text": self.text[:MAX_TEXT_CHARS],
            "output_tokens": self._chunks,
            "thinking": self._thinking,
            "read_ms": self._read_ms,
        }

    def _send(self, now: float) -> None:
        self._sent_at, self._held = now, False
        if self._report is not None:
            self._report(self.snapshot())
