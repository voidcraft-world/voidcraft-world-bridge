"""Routes, the job lifecycle, and the refusals. Every test runs with no Ollama installed.

The worker thread is NOT started (`start_worker=False`); each test runs the
queued job on its own thread with `run_next()`, so nothing here sleeps or races.
"""
from __future__ import annotations

from dataclasses import dataclass

from fake_ollama import FakeOllama

from voidcraft_world_bridge.local_llm import jobs as jobs_mod
from voidcraft_world_bridge.local_llm.plugin import LocalLlmPlugin, _validate_chat


@dataclass
class Context:
    start_dir: str = "."
    log: object = None


def plugin(fake: FakeOllama) -> LocalLlmPlugin:
    return LocalLlmPlugin(Context(), opener=fake, start_worker=False)


CHAT = {"system": "You answer questions.", "user": "What is in this world?", "think": False,
        "format": {"type": "object"}}


def test_routes_are_namespaced():
    assert plugin(FakeOllama()).routes() == [
        "/plugin/local-llm/status", "/plugin/local-llm/job", "/plugin/local-llm/chat",
    ]


def test_status_reports_the_pick_and_hides_embedders():
    fake = FakeOllama(models=[
        {"name": "nomic-embed-text:latest", "details": {"family": "nomic-bert"}},
        {"name": "qwen3.6:35b-a3b", "details": {"family": "qwen35moe"}},
    ])
    status, body = plugin(fake).handle_get("/status", {})
    assert status == 200
    assert body["reachable"] is True
    assert body["picked_model"] == "qwen3.6:35b-a3b"
    assert [m["name"] for m in body["models"]] == ["qwen3.6:35b-a3b"]
    assert body["runtime_version"] == "0.12.3"


def test_status_with_ollama_down_is_200_not_an_error():
    status, body = plugin(FakeOllama(down=True)).handle_get("/status", {})
    assert status == 200
    assert body["reachable"] is False
    assert body["picked_model"] is None
    assert body["error"].startswith("runtime_down")


def test_chat_returns_202_at_once_and_the_job_answers_on_the_worker():
    fake = FakeOllama()
    p = plugin(fake)
    status, body = p.handle_post("/chat", {}, dict(CHAT))
    assert status == 202
    assert body["state"] == jobs_mod.QUEUED
    assert not any(path == "/api/chat" for path, _ in fake.requests), "no generation on the request thread"

    assert p._jobs.run_next() is True
    status, job = p.handle_get("/job", {"id": [body["job_id"]]})
    assert status == 200
    assert job["state"] == jobs_mod.DONE
    assert job["model"] == "qwen3.5:9b"
    assert job["result"]["text"] == '{"answer": "hi", "found": true}'
    assert job["result"]["total_ms"] == 4200

    sent = next(b for path, b in fake.requests if path == "/api/chat")
    assert sent["think"] is False
    assert sent["format"] == {"type": "object"}
    assert sent["options"]["num_ctx"] >= 8192, "context is set explicitly, never Ollama's default"


def test_a_model_without_thinking_is_retried_once_without_the_field():
    fake = FakeOllama(reject_think=True)
    p = plugin(fake)
    _, body = p.handle_post("/chat", {}, dict(CHAT))
    p._jobs.run_next()
    _, job = p.handle_get("/job", {"id": [body["job_id"]]})
    assert job["state"] == jobs_mod.DONE
    chats = [b for path, b in fake.requests if path == "/api/chat"]
    assert len(chats) == 2 and "think" not in chats[1]


def test_a_named_model_that_is_not_installed_fails_the_job_with_the_pull_command():
    p = plugin(FakeOllama())
    _, body = p.handle_post("/chat", {}, {**CHAT, "model": "qwen3.8"})
    p._jobs.run_next()
    _, job = p.handle_get("/job", {"id": [body["job_id"]]})
    assert job["state"] == jobs_mod.FAILED
    assert job["error"]["code"] == "model_not_installed"
    assert "ollama pull qwen3.8" in job["error"]["message"]


def test_runtime_down_fails_the_job_not_the_bridge():
    p = plugin(FakeOllama(down=True))
    _, body = p.handle_post("/chat", {}, dict(CHAT))
    p._jobs.run_next()
    _, job = p.handle_get("/job", {"id": [body["job_id"]]})
    assert job["state"] == jobs_mod.FAILED
    assert job["error"]["code"] == "runtime_down"


def test_the_prompt_is_not_kept_after_the_answer():
    p = plugin(FakeOllama())
    _, body = p.handle_post("/chat", {}, dict(CHAT))
    p._jobs.run_next()
    stored = p._jobs._jobs[body["job_id"]]
    assert stored["request"] == {}


def test_queue_is_bounded():
    p = plugin(FakeOllama())
    for _ in range(jobs_mod.MAX_PENDING):
        assert p.handle_post("/chat", {}, dict(CHAT))[0] == 202
    status, body = p.handle_post("/chat", {}, dict(CHAT))
    assert status == 429
    assert body["error"] == "busy"


def test_bad_bodies_are_400():
    p = plugin(FakeOllama())
    assert p.handle_post("/chat", {}, None)[0] == 400
    assert p.handle_post("/chat", {}, {"system": "x", "user": "  "})[0] == 400
    assert p.handle_post("/chat", {}, {**CHAT, "max_tokens": "lots"})[0] == 400
    assert p.handle_post("/chat", {}, {**CHAT, "format": 7})[0] == 400
    assert p.handle_post("/chat", {}, {**CHAT, "user": "x" * 50_000})[0] == 400


def _sent_chat(body: dict) -> dict:
    fake = FakeOllama()
    p = plugin(fake)
    assert p.handle_post("/chat", {}, body)[0] == 202
    p._jobs.run_next()
    return next(b for path, b in fake.requests if path == "/api/chat")


def test_a_seed_reaches_the_ollama_options():
    sent = _sent_chat({**CHAT, "seed": 42})
    assert sent["options"]["seed"] == 42
    assert "seed" not in sent, "seed is an Ollama OPTION, not a top-level field"
    assert _sent_chat({**CHAT, "seed": 0})["options"]["seed"] == 0, "0 is a seed, not 'unset'"
    assert _sent_chat({**CHAT, "seed": 2**31 - 1})["options"]["seed"] == 2**31 - 1


def test_no_seed_sends_no_seed_key():
    assert "seed" not in _sent_chat(dict(CHAT))["options"]
    assert "seed" not in _sent_chat({**CHAT, "seed": None})["options"]


def test_a_bad_seed_is_400_not_clamped():
    p = plugin(FakeOllama())
    for bad in (True, False, 1.5, 42.0, "42", -1, 2**31, [1], {"seed": 1}):
        status, body = p.handle_post("/chat", {}, {**CHAT, "seed": bad})
        assert status == 400, f"seed={bad!r} should be refused"
        assert "seed" in body["message"]


def test_max_tokens_is_clamped_not_trusted():
    fake = FakeOllama()
    p = plugin(fake)
    p.handle_post("/chat", {}, {**CHAT, "max_tokens": 10**9})
    p._jobs.run_next()
    sent = next(b for path, b in fake.requests if path == "/api/chat")
    assert sent["options"]["num_predict"] == 4096


def test_unknown_routes_and_jobs_are_404():
    p = plugin(FakeOllama())
    assert p.handle_get("/nope", {})[0] == 404
    assert p.handle_get("/job", {"id": ["deadbeef"]})[0] == 404
    assert p.handle_post("/nope", {}, {})[0] == 404


def test_the_plugin_opts_into_a_body_limit_a_real_question_fits():
    # The host's default is 4 KB (server.MAX_BODY_BYTES); a real question is ~15 KB.
    assert LocalLlmPlugin.max_body_bytes >= 256 * 1024


def test_a_tool_conversation_round_trips_and_returns_tool_calls():
    fake = FakeOllama()
    fake.tool_calls = [{"function": {"name": "search", "arguments": {"query": "sourdough"}}}]
    p = plugin(fake)
    tools = [{"type": "function", "function": {"name": "search", "description": "d", "parameters": {"type": "object"}}}]
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "list_worlds", "arguments": {}}]},
        {"role": "tool", "content": "{}", "tool_name": "list_worlds"},
    ]
    status, body = p.handle_post("/chat", {}, {"messages": messages, "tools": tools, "num_ctx": 999_999})
    assert status == 202
    p._jobs.run_next()
    _, job = p.handle_get("/job", {"id": [body["job_id"]]})
    assert job["state"] == "done"
    assert job["result"]["tool_calls"] == [{"name": "search", "arguments": {"query": "sourdough"}}]
    sent = next(b for path, b in fake.requests if path == "/api/chat")
    assert sent["tools"] == tools
    assert sent["options"]["num_ctx"] == 32_768, "num_ctx is clamped"
    assert sent["messages"][2]["tool_calls"] == [{"function": {"name": "list_worlds", "arguments": {}}}]
    assert sent["messages"][3] == {"role": "tool", "content": "{}", "tool_name": "list_worlds"}


def test_a_bad_conversation_is_400():
    p = plugin(FakeOllama())
    assert p.handle_post("/chat", {}, {"messages": []})[0] == 400
    assert p.handle_post("/chat", {}, {"messages": [{"role": "wizard", "content": "x"}]})[0] == 400
    assert p.handle_post("/chat", {}, {"messages": [{"role": "system", "content": "only system"}]})[0] == 400
    assert p.handle_post("/chat", {}, {**CHAT, "tools": [{"type": "nope"}]})[0] == 400


def test_a_tool_entry_that_is_not_an_object_is_a_400_not_a_500():
    # `"nope".get` used to raise inside the handler, which the host reports as a
    # 500 "plugin error". The body came from a browser; its shape is a 400.
    body = {"messages": [{"role": "user", "content": "hi"}], "tools": ["nope"]}
    request, problem = _validate_chat(body)
    assert request == {}
    assert problem is not None and "each tool must be" in problem
