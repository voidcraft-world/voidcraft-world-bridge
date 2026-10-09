"""Finding the model server, speaking OpenAI's shape, and the agent registry.

No model server runs here: `Machine` stands in for urllib's `urlopen` and
answers as whichever servers a test says are "running", by port.
"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from voidcraft_world_bridge.local_llm import openai_compat
from voidcraft_world_bridge.local_llm.agents import AgentRefused
from voidcraft_world_bridge.local_llm.models import matches, pick_model
from voidcraft_world_bridge.local_llm.plugin import LocalLlmPlugin


class _Response:
    def __init__(self, payload) -> None:
        self._raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def _refused(url, code, payload) -> urllib.error.HTTPError:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return urllib.error.HTTPError(url, code, "error", {}, io.BytesIO(body))


class Machine:
    """This machine's loopback ports. Each running server is a handler
    `(path, body) -> payload`, keyed by port; any other port refuses to connect."""

    def __init__(self, **servers) -> None:
        self.servers = {int(port.lstrip("p")): handler for port, handler in servers.items()}
        self.requests: list[tuple[str, dict | None]] = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        port = int(url.split("127.0.0.1:", 1)[1].split("/", 1)[0])
        path = "/" + url.split(f"127.0.0.1:{port}/", 1)[1] if f"{port}/" in url else "/"
        body = json.loads(request.data) if request.data else None
        handler = self.servers.get(port)
        if handler is None:
            raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        self.requests.append((f":{port}{path}", body))
        result = handler(path, body)
        if isinstance(result, urllib.error.HTTPError):
            raise result
        return _Response(result)


def lm_studio(models=("qwen/qwen3.6-35b-a3b", "text-embedding-nomic-embed-text-v1.5"), reply="hi", tool_calls=None):
    def handle(path, body):
        if path == "/v1/models":
            return {"object": "list", "data": [{"id": m, "object": "model"} for m in models]}
        if path == "/v1/chat/completions":
            message = {"role": "assistant", "content": reply}
            if tool_calls:
                message["tool_calls"] = tool_calls
            return {"model": body["model"], "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 7}}
        return _refused(path, 404, {"error": {"message": "Unexpected endpoint"}})
    return handle


def ollama(models=("qwen3.5:9b",)):
    def handle(path, body):
        if path == "/api/tags":
            return {"models": [{"name": m, "details": {"family": "qwen35"}} for m in models]}
        if path == "/api/version":
            return {"version": "0.35.1"}
        if path == "/api/chat":
            return {"model": body["model"], "message": {"role": "assistant", "content": "from ollama"},
                    "prompt_eval_count": 1, "eval_count": 1}
        return _refused(path, 404, {"error": "not found"})
    return handle


def web_app(path, body):
    return _refused(path, 404, b"<html>Cannot GET</html>")


def plugin(machine, **kwargs) -> LocalLlmPlugin:
    return LocalLlmPlugin(object(), opener=machine, start_worker=False, **kwargs)


def ask(p: LocalLlmPlugin, body: dict) -> dict:
    status, job = p.handle_post("/chat", {}, body)
    assert status == 202, job
    p._jobs.run_next()
    return p.handle_get("/job", {"id": [job["job_id"]]})[1]


QUESTION = {"system": "S", "user": "U"}


# --- probing --------------------------------------------------------------------


def test_ollama_wins_when_it_is_running():
    status = plugin(Machine(p11434=ollama(), p1234=lm_studio())).handle_get("/status", {})[1]
    assert (status["runtime"], status["runtime_label"]) == ("ollama", "Ollama")
    assert status["runtime_version"] == "0.35.1"


def test_lm_studio_is_found_with_no_configuration():
    p = plugin(Machine(p1234=lm_studio()))
    status = p.handle_get("/status", {})[1]
    assert status["reachable"] is True
    assert (status["runtime"], status["runtime_label"], status["runtime_url"]) == (
        "openai", "LM Studio", "http://127.0.0.1:1234/v1")
    # The embedder is hidden, and the preference list reads LM Studio's ids.
    assert [m["name"] for m in status["models"]] == ["qwen/qwen3.6-35b-a3b"]
    assert status["picked_model"] == "qwen/qwen3.6-35b-a3b"
    assert ask(p, dict(QUESTION))["result"]["text"] == "hi"


def test_a_web_app_on_a_probed_port_is_passed_over():
    status = plugin(Machine(p1234=web_app, p8080=lm_studio())).handle_get("/status", {})[1]
    assert status["runtime_label"] == "llama.cpp"


def test_nothing_running_names_every_server_it_tried():
    status = plugin(Machine()).handle_get("/status", {})[1]
    assert status["reachable"] is False
    assert status["error"].startswith("runtime_down")
    for where in ("Ollama at http://127.0.0.1:11434", "LM Studio at http://127.0.0.1:1234/v1",
                  "llama.cpp at http://127.0.0.1:8080/v1"):
        assert where in status["error"]


def test_a_configured_url_is_the_only_server_asked(monkeypatch):
    monkeypatch.setenv("VOIDCRAFT_LOCAL_LLM_URL", "http://127.0.0.1:9000/v1/")
    machine = Machine(p11434=ollama(), p9000=lm_studio())
    status = plugin(machine).handle_get("/status", {})[1]
    assert (status["runtime"], status["runtime_url"]) == ("openai", "http://127.0.0.1:9000/v1")
    # Ollama is running on its default port, and is never asked.
    assert machine.requests and all(port_path.startswith(":9000/") for port_path, _ in machine.requests)


def test_a_configured_server_that_refuses_reports_its_own_error(monkeypatch):
    monkeypatch.setenv("VOIDCRAFT_LOCAL_LLM_URL", "http://127.0.0.1:9000/v1")
    status = plugin(Machine(p9000=web_app)).handle_get("/status", {})[1]
    assert status["error"].startswith("runtime_error")


def test_a_custom_ollama_url_still_works(monkeypatch):
    monkeypatch.setenv("VOIDCRAFT_OLLAMA_URL", "http://127.0.0.1:11500")
    status = plugin(Machine(p11500=ollama())).handle_get("/status", {})[1]
    assert (status["runtime"], status["runtime_url"]) == ("ollama", "http://127.0.0.1:11500")


def test_a_missing_model_names_the_fix_for_that_server():
    job = ask(plugin(Machine(p1234=lm_studio())), dict(QUESTION, model="llama-9000"))
    assert job["error"]["code"] == "model_not_installed"
    assert "ollama pull" not in job["error"]["message"]


# --- the OpenAI wire shape -----------------------------------------------------


def test_the_request_is_translated_to_the_openai_shape():
    machine = Machine(p1234=lm_studio())
    ask(plugin(machine), dict(QUESTION, format={"type": "object", "properties": {}}, seed=7, think=False,
                              num_ctx=8192, max_tokens=99))
    _, sent = next(r for r in machine.requests if r[0].endswith("/chat/completions"))
    assert sent["response_format"] == {"type": "json_schema",
                                       "json_schema": {"name": "reply", "schema": {"type": "object", "properties": {}}}}
    assert sent["seed"] == 7 and sent["max_tokens"] == 99 and sent["stream"] is False
    # Not portable — see the module note in openai_compat.py.
    assert "think" not in sent and "num_ctx" not in sent and "options" not in sent


def test_bare_json_format_becomes_an_object_schema():
    machine = Machine(p1234=lm_studio())
    ask(plugin(machine), dict(QUESTION, format="json"))
    _, sent = next(r for r in machine.requests if r[0].endswith("/chat/completions"))
    assert sent["response_format"]["json_schema"]["schema"] == {"type": "object"}


def test_a_leading_think_block_is_removed():
    job = ask(plugin(Machine(p1234=lm_studio(reply="<think>hmm, the user wants…</think>\n{\"a\": 1}"))),
              dict(QUESTION))
    assert job["result"]["text"] == '{"a": 1}'


def test_tool_calls_round_trip_with_ids_and_parsed_arguments():
    calls = [{"id": "x", "type": "function", "function": {"name": "search", "arguments": '{"q": "maps"}'}}]
    machine = Machine(p1234=lm_studio(reply="", tool_calls=calls))
    conversation = {
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "find maps"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "search", "arguments": {"q": "a"}},
                                                                {"name": "open", "arguments": {"id": 1}}]},
            {"role": "tool", "tool_name": "open", "content": "opened"},
            {"role": "tool", "tool_name": "search", "content": "3 hits"},
        ],
        "tools": [{"type": "function", "function": {"name": "search"}}],
    }
    job = ask(plugin(machine), conversation)
    assert job["result"]["tool_calls"] == [{"name": "search", "arguments": {"q": "maps"}}]
    _, sent = next(r for r in machine.requests if r[0].endswith("/chat/completions"))
    assistant = sent["messages"][2]
    assert [c["id"] for c in assistant["tool_calls"]] == ["call_2_0", "call_2_1"]
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"q": "a"}
    # Replies arrive out of order and are matched by tool NAME, not position.
    assert [(m["tool_call_id"], m["content"]) for m in sent["messages"][3:]] == [
        ("call_2_1", "opened"), ("call_2_0", "3 hits")]


def test_a_server_with_no_model_list_is_not_an_openai_server():
    from voidcraft_world_bridge.local_llm.transport import RuntimeRefused
    with pytest.raises(RuntimeRefused):
        openai_compat.list_models("http://127.0.0.1:1234/v1", Machine(p1234=lambda path, body: {"ok": True}))


# --- model ids across naming schemes -------------------------------------------


@pytest.mark.parametrize("preference, name, expected", [
    ("qwen3.6:35b-a3b", "qwen/qwen3.6-35b-a3b", True),
    ("qwen3.6:35b-a3b", "Qwen3.6-35B-A3B-Q4_K_M.gguf", True),
    ("qwen3.6:35b-a3b", "qwen3.6-35b", False),
    ("qwen3.8", "lmstudio-community/qwen3.8-14b", True),
    ("qwen3", "qwen3-30b-a3b", True),
    ("qwen3", "qwen3.6-35b-a3b", False),
])
def test_preferences_match_openai_style_ids(preference, name, expected):
    assert matches(preference, name) is expected


def test_the_strongest_lm_studio_model_is_picked():
    installed = [{"name": "google/gemma4-12b"}, {"name": "qwen/qwen3.6-35b-a3b"}, {"name": "aaa-tiny"}]
    assert pick_model(installed) == "qwen/qwen3.6-35b-a3b"


# --- agent runtimes --------------------------------------------------------------


class EchoAgent:
    """The smallest runtime that satisfies the contract in agents.py."""

    name = "echo"
    default_model = "e1"

    def __init__(self, refuse: bool = False) -> None:
        self.refuse = refuse
        self.ran: list[dict] = []

    def model_error(self, model):
        return None if model in ("e1", "e2") else "model must be e1 or e2"

    def validate_agent(self, raw):
        return ({"tag": raw["tag"]}, None) if isinstance(raw.get("tag"), str) else (None, "agent.tag is required")

    def list_agents(self):
        return [{"id": "echo:1", "runtime": "echo"}]

    def run(self, request):
        self.ran.append(request)
        if self.refuse:
            raise AgentRefused("nope", "refused on purpose")
        return {"text": request["messages"][1]["content"], "tool_calls": [], "model": request["model"]}


AGENT_BODY = dict(QUESTION, agent={"runtime": "echo", "tag": "t"})


def test_a_stock_plugin_has_no_agents():
    p = plugin(Machine())
    assert p.handle_get("/status", {"agents": ["1"]})[1]["agents"] == []
    status, body = p.handle_post("/chat", {}, dict(AGENT_BODY))
    assert status == 400 and "agent.runtime must be one of ()" in body["message"]


def test_a_registered_runtime_answers_on_its_own_queue():
    agent = EchoAgent()
    p = plugin(Machine(), agent_runtimes=[agent])
    status, job = p.handle_post("/chat", {}, dict(AGENT_BODY))
    assert status == 202 and job["model"] == "e1"  # the runtime's default model
    assert p._jobs.run_next() is False  # the local queue never saw it
    assert p._agent_jobs.run_next() is True
    done = p.handle_get("/job", {"id": [job["job_id"]]})[1]
    assert done["result"]["text"] == "U"
    assert agent.ran[0]["agent"] == {"runtime": "echo", "tag": "t"}
    assert p.handle_get("/status", {"agents": ["1"]})[1]["agents"] == [{"id": "echo:1", "runtime": "echo"}]


def test_the_runtime_validates_its_own_fields_and_models():
    p = plugin(Machine(), agent_runtimes=[EchoAgent()])
    assert "agent.tag" in p.handle_post("/chat", {}, dict(QUESTION, agent={"runtime": "echo"}))[1]["message"]
    assert "e1 or e2" in p.handle_post("/chat", {}, dict(AGENT_BODY, model="gpt"))[1]["message"]


def test_a_refusal_fails_the_job_with_the_runtimes_code():
    p = plugin(Machine(), agent_runtimes=[EchoAgent(refuse=True)])
    _, job = p.handle_post("/chat", {}, dict(AGENT_BODY))
    p._agent_jobs.run_next()
    assert p.handle_get("/job", {"id": [job["job_id"]]})[1]["error"] == {
        "code": "nope", "message": "refused on purpose"}
