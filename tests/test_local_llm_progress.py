"""A running job's progress: the streamed reply, the snapshot a page polls, and the store that holds it.

Clocks are injected wherever a test reads time, so nothing here sleeps; the
streams are FakeOllama's NDJSON (`fake_ollama.py`).
"""
from __future__ import annotations

import io
import urllib.error

import pytest
from fake_ollama import COUNTERS, FakeOllama, ndjson, pieces

from voidcraft_world_bridge.local_llm import jobs as jobs_mod
from voidcraft_world_bridge.local_llm import ollama
from voidcraft_world_bridge.local_llm.progress import MAX_TEXT_CHARS, REPORT_INTERVAL_SECONDS, ChatProgress
from voidcraft_world_bridge.local_llm.transport import RuntimeDown, RuntimeRefused, http_ndjson

BASE = "http://127.0.0.1:11434"
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def chat(fake: FakeOllama, **kwargs) -> dict:
    args = {"max_tokens": 100, "temperature": 0.2, "response_format": None, "think": None, **kwargs}
    return ollama.chat(BASE, "qwen3.5:9b", MESSAGES, opener=fake, **args)


# --- ChatProgress -----------------------------------------------------------------


def make_progress() -> tuple[ChatProgress, list[dict], Clock]:
    sent: list[dict] = []
    clock = Clock()
    return ChatProgress(sent.append, clock=clock), sent, clock


def test_reading_is_reported_before_anything_arrives():
    progress, sent, _ = make_progress()
    progress.begin()
    assert sent == [{"phase": "reading", "text": "", "output_tokens": 0, "thinking": False, "read_ms": None}]


def test_the_first_delta_starts_writing_and_stamps_read_ms_once():
    progress, sent, clock = make_progress()
    progress.begin()
    clock.advance(4.2)
    progress.add("Good")
    assert sent[-1] == {"phase": "writing", "text": "Good", "output_tokens": 1, "thinking": False, "read_ms": 4200}
    clock.advance(1.0)
    progress.add(" day")
    progress.finish()
    assert sent[-1]["read_ms"] == 4200, "read_ms is the reading phase; it never moves once set"
    assert sent[-1]["text"] == "Good day" and sent[-1]["output_tokens"] == 2


def test_thinking_is_flagged_until_the_answer_starts():
    progress, sent, clock = make_progress()
    progress.begin()
    clock.advance(0.5)
    progress.add(thinking="Let me")
    assert sent[-1] == {"phase": "writing", "text": "", "output_tokens": 1, "thinking": True, "read_ms": 500}
    progress.add(thinking=" see")
    progress.add("Yes")  # same instant: the throttle would hold it, but the picture changed in kind
    assert sent[-1]["thinking"] is False and sent[-1]["text"] == "Yes"
    assert sent[-1]["output_tokens"] == 3, "thinking chunks are tokens too, as in Ollama's eval_count"


def test_reports_are_throttled_and_the_end_flushes_what_was_held():
    progress, sent, clock = make_progress()
    progress.begin()
    progress.add("a")  # the first delta: sent at once
    for piece in "bcdef":
        clock.advance(0.01)
        progress.add(piece)
    assert len(sent) == 2, "five deltas inside one interval are one write, not five"
    clock.advance(REPORT_INTERVAL_SECONDS)
    progress.add("g")
    assert len(sent) == 3 and sent[-1]["text"] == "abcdefg"
    clock.advance(0.01)
    progress.add("h")
    progress.finish()
    assert len(sent) == 4 and sent[-1]["text"] == "abcdefgh"
    progress.finish()
    assert len(sent) == 4, "nothing held, nothing sent"


def test_a_line_with_no_delta_is_not_a_chunk():
    progress, sent, _ = make_progress()
    progress.add("")
    assert not progress.writing and sent == []


def test_a_tool_call_chunk_counts_and_starts_writing():
    progress, sent, _ = make_progress()
    progress.add(tool_calls=True)
    assert progress.writing and sent[-1]["output_tokens"] == 1 and sent[-1]["text"] == ""


def test_the_snapshot_text_is_capped_but_the_answer_is_not():
    progress, sent, _ = make_progress()
    progress.add("x" * (MAX_TEXT_CHARS + 10))
    assert len(sent[-1]["text"]) == MAX_TEXT_CHARS
    assert len(progress.text) == MAX_TEXT_CHARS + 10


def test_with_no_sink_it_only_accumulates():
    progress = ChatProgress()
    progress.begin()
    progress.add("a")
    progress.add("b")
    progress.finish()
    assert progress.text == "ab"


# --- http_ndjson ------------------------------------------------------------------


class Body:
    """A streamed reply whose reads can fail partway."""

    def __init__(self, lines: bytes, fail_after: int | None = None) -> None:
        self._body = io.BytesIO(lines)
        self._reads = 0
        self._fail_after = fail_after

    def readline(self, limit: int = -1) -> bytes:
        self._reads += 1
        if self._fail_after is not None and self._reads > self._fail_after:
            raise TimeoutError("timed out")
        return self._body.readline(limit)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def read_all(body: Body, clock=None) -> list[dict]:
    kwargs = {"clock": clock} if clock else {}
    with http_ndjson(f"{BASE}/api/chat", {}, 300.0, lambda request, timeout: body, **kwargs) as lines:
        return list(lines)


def test_lines_are_parsed_as_they_come_and_blank_ones_skipped():
    assert read_all(Body(b'{"a": 1}\n\n{"b": 2}\n')) == [{"a": 1}, {"b": 2}]


def test_an_error_line_is_a_refusal():
    with pytest.raises(RuntimeRefused, match="model runner has unexpectedly stopped"):
        read_all(Body(ndjson([{"a": 1}, {"error": "model runner has unexpectedly stopped"}])))


def test_a_stalled_socket_is_runtime_down():
    with pytest.raises(RuntimeDown, match="broke off: timed out"):
        read_all(Body(ndjson([{"a": 1}, {"b": 2}]), fail_after=1))


def test_a_reply_that_keeps_trickling_ends_at_the_deadline():
    clock = Clock()

    class Slow(Body):
        def readline(self, limit=-1):
            clock.advance(200)
            return super().readline(limit)

    with pytest.raises(RuntimeDown, match="within 300 s"):
        read_all(Slow(ndjson([{"a": 1}, {"b": 2}])), clock)


def test_a_line_that_is_not_an_object_or_too_long_is_refused():
    with pytest.raises(RuntimeRefused, match="not a JSON object"):
        read_all(Body(b"[1, 2]\n"))
    with pytest.raises(RuntimeRefused, match="unparsable stream line"):
        read_all(Body(b"{nope\n"))
    with pytest.raises(RuntimeRefused, match="longer than"):
        read_all(Body(b'{"a": "' + b"x" * (1024 * 1024 + 8) + b'"}\n'))


def test_opening_fails_the_same_two_ways_as_a_one_shot_call():
    def refused(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "x", {}, io.BytesIO(b'{"error": "model not found"}'))

    def down(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))

    for opener, error in ((refused, RuntimeRefused), (down, RuntimeDown)):
        with pytest.raises(error), http_ndjson(f"{BASE}/api/chat", {}, 1.0, opener):
            pass


# --- ollama.chat, streamed --------------------------------------------------------


def test_a_streamed_answer_has_the_one_shot_result_shape():
    fake = FakeOllama(chat_reply="Paris is sunny.", thinking="Weather question.")
    fake.tool_calls = [{"id": "call_1", "function": {"index": 0, "name": "get_weather",
                                                     "arguments": {"city": "Paris"}}}]
    result = chat(fake, response_format={"type": "object"}, tools=[{"type": "function"}])
    assert result == {
        "text": "Paris is sunny.",
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
        "model": "qwen3.5:9b",
        "prompt_tokens": COUNTERS["prompt_eval_count"],
        "output_tokens": COUNTERS["eval_count"],
        "total_ms": 4200,
        "load_ms": 1000,
        "prompt_ms": 600,
        "output_ms": 2000,
    }
    (_, sent), = fake.requests
    assert sent["stream"] is True
    assert sent["format"] == {"type": "object"} and sent["tools"] == [{"type": "function"}]


def test_progress_goes_reading_thinking_writing_and_ends_whole():
    fake = FakeOllama(chat_reply="The answer is 42.", thinking="Hmm, think.")
    sent: list[dict] = []
    result = chat(fake, on_progress=sent.append)
    assert sent[0]["phase"] == "reading" and sent[0]["read_ms"] is None
    assert sent[1]["phase"] == "writing" and sent[1]["thinking"] is True and sent[1]["text"] == ""
    assert isinstance(sent[1]["read_ms"], int)
    answering = next(s for s in sent if s["text"])
    assert answering["thinking"] is False
    assert sent[-1]["text"] == result["text"] == "The answer is 42."
    assert sent[-1]["output_tokens"] == len(pieces("Hmm, think.")) + len(pieces("The answer is 42."))
    assert {s["read_ms"] for s in sent[1:]} == {sent[1]["read_ms"]}


def test_the_connection_is_closed_when_the_answer_is_read():
    fake = FakeOllama()
    chat(fake)
    assert all(response.closed for response in fake.responses)


def test_a_think_refusal_in_the_stream_is_retried_once_without_think():
    fake = FakeOllama(reject_think_in_stream=True)
    sent: list[dict] = []
    result = chat(fake, think=False, on_progress=sent.append)
    assert result["text"] == '{"answer": "hi", "found": true}'
    chats = [b for path, b in fake.requests if path == "/api/chat"]
    assert len(chats) == 2 and "think" in chats[0] and "think" not in chats[1]
    assert sent[0]["phase"] == "reading"


def test_a_think_refusal_over_http_is_retried_once_without_think():
    fake = FakeOllama(reject_think=True)
    assert chat(fake, think="low")["text"] == '{"answer": "hi", "found": true}'
    assert len([b for path, b in fake.requests if path == "/api/chat"]) == 2


def test_an_error_after_the_answer_began_is_never_retried():
    # Retrying would send the question twice and splice two answers; only a refusal up front is retried.
    fake = FakeOllama(stream_error="thinking broke the runner", stream_error_after=2)
    with pytest.raises(RuntimeRefused, match="thinking broke the runner"):
        chat(fake, think=True)
    assert len([b for path, b in fake.requests if path == "/api/chat"]) == 1


def test_a_reply_cut_off_before_its_final_line_is_runtime_down():
    with pytest.raises(RuntimeDown, match="ended before"):
        chat(FakeOllama(cut_off=True))


# --- the job store ------------------------------------------------------------------


def test_a_reporting_runner_publishes_progress_only_while_running():
    seen: dict = {}

    def runner(request, report):
        progress = {"phase": "writing", "text": "par", "output_tokens": 1, "thinking": False, "read_ms": 9}
        report(progress)
        progress["text"] = "changed after the report"  # the store kept a copy
        seen["running"] = store.get(job_id)
        return {"text": "partial answer"}

    store = jobs_mod.JobStore(runner, reports_progress=True, start_worker=False)
    job_id = store.submit({})["id"]
    assert store.get(job_id)["progress"] is None, "queued: nothing yet"
    store.run_next()
    assert seen["running"]["state"] == jobs_mod.RUNNING
    assert seen["running"]["progress"] == {"phase": "writing", "text": "par", "output_tokens": 1,
                                           "thinking": False, "read_ms": 9}
    done = store.get(job_id)
    assert done["state"] == jobs_mod.DONE and done["progress"] is None
    assert done["result"] == {"text": "partial answer"}


def test_a_failed_job_drops_its_progress():
    def runner(request, report):
        report({"phase": "reading"})
        raise jobs_mod.JobFailure("runtime_down", "gone")

    store = jobs_mod.JobStore(runner, reports_progress=True, start_worker=False)
    job_id = store.submit({})["id"]
    store.run_next()
    job = store.get(job_id)
    assert job["state"] == jobs_mod.FAILED and job["progress"] is None


def test_a_report_after_the_job_finished_is_ignored():
    sinks = []

    def runner(request, report):
        sinks.append(report)
        return {}

    store = jobs_mod.JobStore(runner, reports_progress=True, start_worker=False)
    job_id = store.submit({})["id"]
    store.run_next()
    sinks[0]({"phase": "writing"})
    assert store._jobs[job_id]["progress"] is None


def test_a_one_argument_runner_still_works_and_reports_nothing():
    seen: dict = {}

    def runner(request):
        seen["running"] = store.get(job_id)
        return {"text": "ok"}

    store = jobs_mod.JobStore(runner, start_worker=False)
    job_id = store.submit({})["id"]
    store.run_next()
    assert seen["running"]["state"] == jobs_mod.RUNNING and seen["running"]["progress"] is None
    assert store.get(job_id)["result"] == {"text": "ok"}
