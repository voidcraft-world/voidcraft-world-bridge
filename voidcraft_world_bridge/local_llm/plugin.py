"""The plugin object the bridge mounts: GET /status, GET /job, POST /chat.

Local intelligence for voidcraft.world, on this machine. It is deliberately
generic — system + user prompt in, text out — so any caller can use it: a
question about a world, an Arena commander's turn. The prompt is always built by
the caller, never here, so a local model and a cloud model answer the SAME
prompt and comparing them compares models, not prompts.

Why a page goes through the bridge at all instead of calling the model server
directly: the browser will not let it. A page on voidcraft.world reaching
`localhost` hits CORS and Chrome's Local Network Access, and each model server
configures those differently. The bridge is one door, with one allowlist.

Which server answers — Ollama, LM Studio, llama.cpp, any OpenAI-compatible one —
is found by probing (`runtimes.py`). A body with an `agent` field goes to an
agent runtime the host registered instead (`agents.py`).

── WHAT THIS PLUGIN WILL NOT DO ─────────────────────────────────────────────
- PULL a model. A named model must already be installed (`models.py`).
- KEEP a prompt. A job drops its request once answered (`jobs.py`).
- Reach anything but a model server on this machine (or the one URL the user
  configured) — or, for an agent, the runtime the host registered.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Iterable

from voidcraft_world_bridge.local_llm import jobs as jobs_mod
from voidcraft_world_bridge.local_llm import models as models_mod
from voidcraft_world_bridge.local_llm import runtimes as runtimes_mod
from voidcraft_world_bridge.local_llm.agents import AgentRefused
from voidcraft_world_bridge.local_llm.transport import RuntimeDown, RuntimeRefused

PLUGIN_NAME = "local-llm"

STATUS_VERSION = 1
"""The `/status` shape. Additive-only, like the bridge's own `/snapshot`."""

PIN_ENV = "VOIDCRAFT_LOCAL_LLM_MODEL"

MAX_SYSTEM_CHARS = 16_000
MAX_USER_CHARS = 48_000
MAX_TOKENS_CEILING = 4096
DEFAULT_MAX_TOKENS = 800

MAX_MESSAGES = 48
MAX_CONVERSATION_CHARS = 160_000
MAX_TOOLS = 24
MAX_TOOLS_CHARS = 64_000
MAX_NUM_CTX = 32_768
"""A multi-turn tool conversation: one system message, the question, and a
tool call + result per step. Bounded because the body arrives from a browser."""

MAX_SEED = 2**31 - 1
"""A fixed seed makes one request reproducible — an Arena match sends one per
turn so a replay asks the same question the same way. Ollama treats a negative
seed as "random", so 0..2^31-1 is the whole useful range, and it fits every
runtime's seed type and a JS number."""

MESSAGE_ROLES = ("system", "user", "assistant", "tool")

MODELS_TTL_SECONDS = 10.0
"""How long one model-list answer (and the probe that found its server) is
reused. The page polls `/status` when it opens and a job resolves its model on
every run; neither needs a fresher list than "installed in the last ten
seconds"."""

UNKNOWN_ROUTE = {"error": "unknown route — GET /status, GET /job?id=, POST /chat"}


class LocalLlmPlugin:
    name = PLUGIN_NAME

    max_body_bytes = 512 * 1024
    """The host caps plugin POSTs at 4 KB unless a plugin opts in
    (`plugin_body_limit`). One question carrying a world's sources is ~15 KB
    and a tool conversation ~100–160 KB, so this plugin opts in — WITHOUT it
    every real question is a 400 "bad content length" before the plugin ever
    sees it. The per-field limits in `_validate_chat` stay the real bound; this
    only lets the body in."""

    def __init__(self, context, *, opener=None, start_worker: bool = True,
                 agent_runtimes: Iterable[object] = ()) -> None:
        self._context = context
        self._opener = opener
        self._models_lock = threading.Lock()
        # (stamp, runtime, models, runtime_version, error) — see _installed.
        self._models_cache: tuple = (0.0, None, None, None, None)
        self._jobs = jobs_mod.JobStore(self._run_chat, start_worker=start_worker)
        # Agents get their own queue and worker: a remote turn shares no memory
        # bandwidth with the local model, so it must not wait behind a local
        # generation.
        self._agent_runtimes = {runtime.name: runtime for runtime in agent_runtimes}
        self._agent_jobs = jobs_mod.JobStore(self._run_agent, start_worker=start_worker)

    def routes(self) -> list[str]:
        base = f"/plugin/{self.name}"
        return [f"{base}/status", f"{base}/job", f"{base}/chat"]

    # -- GET ------------------------------------------------------------------
    def handle_get(self, subpath: str, query: dict[str, list[str]]) -> tuple:
        if subpath == "/status":
            status = self._status()
            if (query.get("agents") or [""])[0] == "1":
                status["agents"] = self._agents()
            return 200, status
        if subpath == "/job":
            job_id = (query.get("id") or [""])[0]
            job = (self._jobs.get(job_id) or self._agent_jobs.get(job_id)) if job_id else None
            if job is None:
                return 404, {"error": "unknown_job"}
            return 200, job
        return 404, dict(UNKNOWN_ROUTE)

    # -- POST -----------------------------------------------------------------
    def handle_post(self, subpath: str, query: dict[str, list[str]], body: dict | None) -> tuple:
        if subpath != "/chat":
            return 404, dict(UNKNOWN_ROUTE)
        request, problem = _validate_chat(body, self._agent_runtimes)
        if problem:
            return 400, {"error": "bad_request", "message": problem}
        store = self._agent_jobs if request.get("agent") else self._jobs
        try:
            job = store.submit(request)
        except jobs_mod.QueueFull:
            waiting_for = "the agent" if request.get("agent") else "the local model"
            return 429, {"error": "busy",
                         "message": f"{jobs_mod.MAX_PENDING} questions are already waiting for {waiting_for}"}
        return 202, {"job_id": job["id"], **job}

    # -- internals ------------------------------------------------------------
    def _installed(self) -> tuple:
        """(runtime, models, runtime_version, error) — the first server that
        answers, cached for MODELS_TTL_SECONDS. `models` is None when none did.

        A server that answers with an error (a web app squatting on a probed
        port, an endpoint that is not a model list) is passed over like one
        that is down; the error is reported only when it is the ONE server the
        user configured, since then it is the answer.
        """
        now = time.monotonic()
        with self._models_lock:
            stamp, runtime, models, runtime_version, error = self._models_cache
            if models is not None and now - stamp < MODELS_TTL_SECONDS:
                return runtime, models, runtime_version, error
        candidates = runtimes_mod.candidates()
        found = None
        problems = []
        for candidate in candidates:
            try:
                found = (candidate, candidate.list_models(self._opener), candidate.version(self._opener), None)
                break
            except RuntimeDown as err:
                problems.append(("runtime_down", candidate, str(err)))
            except RuntimeRefused as err:
                problems.append(("runtime_error", candidate, err.message))
        if found is None:
            if len(candidates) == 1:
                code, _, detail = problems[0]
                error = f"{code}: {detail}"
            else:
                tried = ", ".join(candidate.describe() for _, candidate, _ in problems)
                error = f"runtime_down: no local model server answered (tried {tried})"
            found = (candidates[0], None, None, error)
        with self._models_lock:
            self._models_cache = (now, *found)
        return found

    def _status(self) -> dict:
        runtime, models, runtime_version, error = self._installed()
        pinned = os.environ.get(PIN_ENV) or None
        picked = models_mod.pick_model(models or [], pinned)
        return {
            "version": STATUS_VERSION,
            "plugin": self.name,
            "runtime": runtime.kind,
            "runtime_label": runtime.label,
            "runtime_url": runtime.base,
            "reachable": models is not None,
            "runtime_version": runtime_version,
            "error": error,
            "models": [m for m in (models or []) if models_mod.is_chat_model(m)],
            "picked_model": picked,
            "pinned_model": pinned,
            "preference": list(models_mod.PREFERENCE),
            **self._jobs.counts(),
        }

    def _agents(self) -> list[dict]:
        """Every agent each registered runtime offers (`agents.py`)."""
        rows: list[dict] = []
        for runtime in self._agent_runtimes.values():
            rows.extend(runtime.list_agents())
        return rows

    def _run_agent(self, request: dict) -> dict:
        runtime = self._agent_runtimes[request["agent"]["runtime"]]
        try:
            return runtime.run(request)
        except AgentRefused as refused:
            raise jobs_mod.JobFailure(refused.code, refused.message) from refused

    def _run_chat(self, request: dict) -> dict:
        runtime, models, _, error = self._installed()
        if models is None:
            raise jobs_mod.JobFailure("runtime_down", error or f"{runtime.label} is not answering")
        model, problem = models_mod.resolve_requested(request.get("model"), models,
                                                      os.environ.get(PIN_ENV) or None)
        if problem == "model_not_installed":
            raise jobs_mod.JobFailure(problem, runtime.missing_model_hint(str(request.get("model"))))
        if problem or not model:
            raise jobs_mod.JobFailure("no_model", runtime.no_model_hint)
        try:
            result = runtime.chat(model, request["messages"],
                                  max_tokens=request["max_tokens"], temperature=request["temperature"],
                                  response_format=request.get("format"), think=request.get("think"),
                                  tools=request.get("tools"), num_ctx=request.get("num_ctx"),
                                  seed=request.get("seed"), opener=self._opener)
        except RuntimeDown as err:
            raise jobs_mod.JobFailure("runtime_down", f"{runtime.label} stopped answering: {err}") from err
        except RuntimeRefused as err:
            raise jobs_mod.JobFailure("runtime_error", err.message) from err
        return result


def _validate_messages(raw: object) -> tuple[list[dict], str | None]:
    """A multi-turn conversation: roles checked, sizes bounded, tool calls passed through as Ollama takes them."""
    if not isinstance(raw, list) or not raw or len(raw) > MAX_MESSAGES:
        return [], f"messages must be a list of 1–{MAX_MESSAGES} turns"
    messages, total = [], 0
    for turn in raw:
        if not isinstance(turn, dict) or turn.get("role") not in MESSAGE_ROLES:
            return [], f"each message needs a role in {MESSAGE_ROLES}"
        content = turn.get("content", "")
        if not isinstance(content, str):
            return [], "message content must be a string"
        total += len(content)
        out = {"role": turn["role"], "content": content}
        if turn["role"] == "assistant" and isinstance(turn.get("tool_calls"), list):
            out["tool_calls"] = [
                {"function": {"name": str(c.get("name", "")), "arguments": c.get("arguments") if isinstance(c.get("arguments"), dict) else {}}}
                for c in turn["tool_calls"] if isinstance(c, dict)
            ]
        if turn["role"] == "tool" and isinstance(turn.get("tool_name"), str):
            out["tool_name"] = turn["tool_name"]
        messages.append(out)
    if total > MAX_CONVERSATION_CHARS:
        return [], f"conversation too long (≤ {MAX_CONVERSATION_CHARS} chars)"
    if not any(m["role"] == "user" and m["content"].strip() for m in messages):
        return [], "the conversation needs a non-empty user message"
    return messages, None


def _validate_tools(raw: object) -> tuple[list[dict] | None, str | None]:
    if raw is None:
        return None, None
    if not isinstance(raw, list) or len(raw) > MAX_TOOLS or len(json.dumps(raw)) > MAX_TOOLS_CHARS:
        return None, f"tools must be a list of ≤ {MAX_TOOLS} definitions (≤ {MAX_TOOLS_CHARS} chars)"
    for tool in raw:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(fn, dict) \
                or not isinstance(fn.get("name"), str):
            return None, "each tool must be {type: 'function', function: {name, description, parameters}}"
    return raw, None


def _validate_agent(raw: object, messages: list[dict], tools: list[dict] | None,
                    agent_runtimes: dict[str, object]) -> tuple[dict | None, object, str | None]:
    """An agent turn: `{runtime, …}` naming a registered runtime, one system +
    one user message and no tools. Returns (agent, runtime, problem); the
    runtime checks its own fields (`agents.py`)."""
    if not isinstance(raw, dict):
        return None, None, "agent must be an object {runtime, …}"
    runtime = agent_runtimes.get(raw.get("runtime"))
    if runtime is None:
        return None, None, f"agent.runtime must be one of {tuple(agent_runtimes)}"
    agent, problem = runtime.validate_agent(raw)
    if problem:
        return None, None, problem
    if tools or [m["role"] for m in messages] != ["system", "user"]:
        return None, None, "an agent takes one system and one user message, and no tools"
    return {"runtime": runtime.name, **(agent or {})}, runtime, None


def _validate_chat(body: dict | None, agent_runtimes: dict[str, object] | None = None) -> tuple[dict, str | None]:
    """Clamp and type-check a POST /chat body. The body arrives from a browser.

    Two shapes: `system` + `user` (one question), or `messages` (a conversation,
    optionally with `tools`). Both become `messages` for the adapter.
    """
    if not isinstance(body, dict):
        return {}, "body must be a JSON object"
    if "messages" in body:
        messages, problem = _validate_messages(body.get("messages"))
        if problem:
            return {}, problem
    else:
        system = body.get("system")
        user = body.get("user")
        if not isinstance(system, str) or not isinstance(user, str) or not user.strip():
            return {}, "system and user must be strings, and user must not be empty"
        if len(system) > MAX_SYSTEM_CHARS or len(user) > MAX_USER_CHARS:
            return {}, f"prompt too long (system ≤ {MAX_SYSTEM_CHARS}, user ≤ {MAX_USER_CHARS} chars)"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    tools, problem = _validate_tools(body.get("tools"))
    if problem:
        return {}, problem

    num_ctx = body.get("num_ctx")
    if num_ctx is not None and (isinstance(num_ctx, bool) or not isinstance(num_ctx, int)):
        return {}, "num_ctx must be an integer"
    num_ctx = max(2048, min(num_ctx, MAX_NUM_CTX)) if num_ctx else None

    model = body.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip() or len(model) > 128):
        return {}, "model must be a non-empty string"

    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        return {}, "max_tokens must be an integer"
    max_tokens = max(1, min(max_tokens, MAX_TOKENS_CEILING))

    temperature = body.get("temperature", 0.2)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        return {}, "temperature must be a number"
    temperature = max(0.0, min(float(temperature), 1.5))

    response_format = body.get("format")
    if response_format is not None and not (response_format == "json" or isinstance(response_format, dict)):
        return {}, 'format must be "json" or a JSON schema object'

    think = body.get("think")
    if think is not None and not isinstance(think, bool):
        return {}, "think must be a boolean"

    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED):
        return {}, f"seed must be an integer from 0 to {MAX_SEED}"

    agent = None
    if body.get("agent") is not None:
        agent, runtime, problem = _validate_agent(body.get("agent"), messages, tools, agent_runtimes or {})
        if problem:
            return {}, problem
        model = model if model is not None else runtime.default_model
        problem = runtime.model_error(model.strip() if isinstance(model, str) else model)
        if problem:
            return {}, problem

    return {
        "agent": agent,
        "messages": messages,
        "tools": tools,
        "num_ctx": num_ctx,
        "model": model.strip() if isinstance(model, str) else None,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "format": response_format,
        "think": think,
        "seed": seed,
    }, None
