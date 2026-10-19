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
agent runtime the host registered instead (`agents.py`). Other machines the
user named (`hosts.py`) answer too, each on its own queue, reached over SSH
(`tunnel.py`); `/status` lists them under `hosts`, and `POST /hosts` edits them.

── WHAT THIS PLUGIN WILL NOT DO ─────────────────────────────────────────────
- PULL a model. A named model must already be installed (`models.py`).
- KEEP a prompt. A job drops its request once answered (`jobs.py`).
- Reach anything but a model server on this machine (or the one URL the user
  configured), a model server on another machine's loopback through an SSH
  alias the user configured — or, for an agent, the runtime the host registered.
"""
from __future__ import annotations

import functools
import json
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import NamedTuple

from voidcraft_world_bridge.local_llm import hosts as hosts_mod
from voidcraft_world_bridge.local_llm import jobs as jobs_mod
from voidcraft_world_bridge.local_llm import models as models_mod
from voidcraft_world_bridge.local_llm import runtimes as runtimes_mod
from voidcraft_world_bridge.local_llm.agents import AgentRefused, AgentRuntime
from voidcraft_world_bridge.local_llm.transport import RuntimeDown, RuntimeRefused
from voidcraft_world_bridge.local_llm.tunnel import NONE, STARTING, SshTunnel
from voidcraft_world_bridge.policy import CallerPolicy
from voidcraft_world_bridge.shape import typed

PLUGIN_NAME = "local-llm"

STATUS_VERSION = 1
"""The `/status` shape. Additive-only, like the bridge's own `/snapshot`."""

PIN_ENV = "VOIDCRAFT_LOCAL_LLM_MODEL"

MAX_SYSTEM_CHARS = 16_000
MAX_USER_CHARS = 48_000
MAX_TOKENS_CEILING = 4096
# `think` as a reasoning effort, for a model that reads levels and cannot turn reasoning
# off (gpt-oss ignores `false` and reasons at medium). Passed to Ollama as given.
THINK_LEVELS = ("low", "medium", "high")
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

UNKNOWN_ROUTE = {"error": "unknown route — GET /status, GET /job?id=, POST /chat, POST /hosts"}


class Installed(NamedTuple):
    """What the probe found: the first server that answered, or the first
    candidate with `models` None and `error` saying why (`_installed`)."""

    runtime: runtimes_mod.Runtime
    models: list[dict] | None
    runtime_version: str | None
    error: str | None


SPEED_SAMPLES = 8
"""Recent answers per host that `/status` averages into tok/s. Few enough that a
model swap shows within a match, enough that one cold start does not dominate."""

TUNNEL_STARTING_MESSAGE = "connecting over SSH"

HOSTS_POLICY = CallerPolicy(loopback_only=True, reject_proxy=True, reject_no_cors=True,
                            subject="host list")
"""Adding a host makes this machine open SSH sessions, so the route that edits
the list answers only on this machine's own loopback name, never through a
proxy, and never to a no-cors browser request. The host's Origin guard still
runs first (`server.py`)."""


class Speed:
    """Rolling prefill and output tok/s for one host, from the answers it gave."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: deque[tuple[int, float | None, int, float | None]] = deque(maxlen=SPEED_SAMPLES)

    def record(self, timings: dict | None) -> None:
        if timings:
            with self._lock:
                self._samples.append((timings["prompt_tokens"], timings["prompt_s"],
                                      timings["output_tokens"], timings["output_s"]))

    def rates(self) -> tuple[float | None, float | None]:
        """(output tok/s, prefill tok/s); None until a sample has that half."""
        with self._lock:
            samples = list(self._samples)
        return (_rate([(o, os_) for _, _, o, os_ in samples]),
                _rate([(p, ps) for p, ps, _, _ in samples]))


def _rate(pairs: list[tuple[int, float | None]]) -> float | None:
    tokens = sum(t for t, s in pairs if s and t > 0)
    seconds = sum(s for t, s in pairs if s and t > 0)
    return round(tokens / seconds, 1) if seconds > 0 else None


def _timings(result: dict) -> dict | None:
    """What one answer says about its host's speed. Ollama and llama-server
    report prefill and output separately; any other server only its wall time,
    which stands in for output (a slight under-estimate of the rate)."""
    prompt_ms, output_ms = result.get("prompt_ms"), result.get("output_ms")
    if output_ms is None:
        output_ms = result.get("total_ms")
    if output_ms is None:
        return None
    return {
        "prompt_tokens": int(result.get("prompt_tokens") or 0),
        "prompt_s": None if prompt_ms is None else prompt_ms / 1000,
        "output_tokens": int(result.get("output_tokens") or 0),
        "output_s": output_ms / 1000,
    }


class Lane:
    """One machine that answers questions: its queue, its speed, and — for
    another machine — the host entry and the tunnel that reaches it."""

    def __init__(self, name: str, store: jobs_mod.JobStore, remote: hosts_mod.RemoteHost | None = None,
                 tunnel: SshTunnel | None = None) -> None:
        self.name, self.store, self.remote, self.tunnel = name, store, remote, tunnel
        self.speed = Speed()
        self.lock = threading.Lock()
        # (stamp, tunnel port the list was read through, what it found).
        self.cache: tuple[float, int | None, Installed | None] = (0.0, None, None)

    @property
    def is_local(self) -> bool:
        return self.remote is None


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
                 agent_runtimes: Iterable[AgentRuntime] = (),
                 hosts_path: Path | None = None, ssh_config_path: Path | None = None,
                 tunnel_factory: Callable[[hosts_mod.RemoteHost], SshTunnel] | None = None) -> None:
        self._context = context
        self._opener = opener
        self._start_worker = start_worker
        self._hosts_path = hosts_path
        self._ssh_config_path = ssh_config_path
        self._tunnel_factory = tunnel_factory or _ssh_tunnel
        self._models_lock = threading.Lock()
        # (stamp, what the probe found) — see _installed.
        self._models_cache: tuple[float, Installed | None] = (0.0, None)
        self._jobs = jobs_mod.JobStore(functools.partial(self._run_chat_on, hosts_mod.LOCAL),
                                       start_worker=start_worker, host=hosts_mod.LOCAL)
        self._lanes_lock = threading.Lock()
        self._lanes: dict[str, Lane] = {hosts_mod.LOCAL: Lane(hosts_mod.LOCAL, self._jobs)}
        # A removed host's store, kept so its finished jobs still answer /job.
        self._retired: list[jobs_mod.JobStore] = []
        for remote in hosts_mod.load_hosts(hosts_path):
            self._add_lane(remote)
        # Agents get their own queue and worker: a remote turn shares no memory
        # bandwidth with the local model, so it must not wait behind a local
        # generation.
        self._agent_runtimes: dict[str, AgentRuntime] = {runtime.name: runtime for runtime in agent_runtimes}
        self._agent_jobs = jobs_mod.JobStore(self._run_agent, start_worker=start_worker, host=hosts_mod.LOCAL)

    def close(self) -> None:
        for lane in self._lane_list():
            lane.store.close()
            if lane.tunnel is not None:
                lane.tunnel.close()
        self._agent_jobs.close()

    def routes(self) -> list[str]:
        base = f"/plugin/{self.name}"
        return [f"{base}/status", f"{base}/job", f"{base}/chat", f"{base}/hosts"]

    # -- dispatch -------------------------------------------------------------
    def handle_request(self, method: str, subpath: str, query: dict[str, list[str]],
                       body: dict | None, headers: dict[str, str]) -> tuple:
        """The host prefers this entry point; it is here only so `/hosts` can read
        the request's headers (`HOSTS_POLICY`). Every other route is unchanged."""
        if subpath == "/hosts":
            refusal = HOSTS_POLICY.refusal(headers)
            if refusal is not None:
                return 403, {"error": refusal.code, "message": refusal.message}
        if method == "GET":
            return self.handle_get(subpath, query)
        return self.handle_post(subpath, query, body)

    # -- GET ------------------------------------------------------------------
    def handle_get(self, subpath: str, query: dict[str, list[str]]) -> tuple:
        if subpath == "/status":
            status = self._status()
            if (query.get("agents") or [""])[0] == "1":
                status["agents"] = self._agents()
            return 200, status
        if subpath == "/job":
            job_id = (query.get("id") or [""])[0]
            job = self._find_job(job_id) if job_id else None
            if job is None:
                return 404, {"error": "unknown_job"}
            return 200, job
        return 404, dict(UNKNOWN_ROUTE)

    # -- POST -----------------------------------------------------------------
    def handle_post(self, subpath: str, query: dict[str, list[str]],  # noqa: ARG002 — the handler signature is the contract
                    body: dict | None) -> tuple:
        if subpath == "/hosts":
            return self._edit_hosts(body)
        if subpath != "/chat":
            return 404, dict(UNKNOWN_ROUTE)
        request, problem = _validate_chat(body, self._agent_runtimes)
        if problem:
            return 400, {"error": "bad_request", "message": problem}
        if request.get("agent"):
            store, waiting_for = self._agent_jobs, "the agent"
        else:
            lane, refusal = self._route(request)
            if lane is None:
                return refusal
            store = lane.store
            waiting_for = "the local model" if lane.is_local else f"the model on {lane.name}"
        try:
            job = store.submit(request)
        except jobs_mod.QueueFull:
            return 429, {"error": "busy",
                         "message": f"{jobs_mod.MAX_PENDING} questions are already waiting for {waiting_for}"}
        return 202, {"job_id": job["id"], **job}

    # -- hosts ----------------------------------------------------------------
    def _lane_list(self) -> list[Lane]:
        with self._lanes_lock:
            return list(self._lanes.values())

    def _add_lane(self, remote: hosts_mod.RemoteHost) -> Lane:
        store = jobs_mod.JobStore(functools.partial(self._run_chat_on, remote.name),
                                  start_worker=self._start_worker, host=remote.name)
        lane = Lane(remote.name, store, remote, self._tunnel_factory(remote))
        with self._lanes_lock:
            self._lanes[remote.name] = lane
        return lane

    def _remotes(self) -> list[hosts_mod.RemoteHost]:
        return [lane.remote for lane in self._lane_list() if lane.remote is not None]

    def _edit_hosts(self, body: dict | None) -> tuple:
        """`{"action": "add", "ssh": alias, "name"?, "kind"?, "port"?}` or `{"action": "remove", "name"}`."""
        body = body if isinstance(body, dict) else {}
        action = body.get("action")
        if action == "add":
            alias = body.get("ssh")
            problem = hosts_mod.alias_problem(alias) or hosts_mod.alias_problem(body.get("name", alias))
            if problem:
                return 400, {"error": "bad_alias", "message": problem}
            if alias not in hosts_mod.configured_aliases(self._ssh_config_path):
                return 400, {"error": "bad_alias",
                             "message": f"no `Host {alias}` line in your SSH config — add one, then run "
                                        f"`ssh {alias}` once so its key and host key are set up"}
            remote = hosts_mod.parse_host({k: body[k] for k in ("ssh", "name", "kind", "port") if k in body})
            if remote is None:
                return 400, {"error": "bad_request", "message": f"kind must be one of {hosts_mod.KINDS}, "
                                                                "port 1–65535"}
            with self._lanes_lock:
                existing = self._lanes.get(remote.name)
            if existing is not None and existing.remote != remote:
                return 400, {"error": "host_exists", "message": f"a host named {remote.name} already exists"}
            if existing is None:
                if len(self._remotes()) >= hosts_mod.MAX_HOSTS:
                    return 400, {"error": "too_many_hosts", "message": f"at most {hosts_mod.MAX_HOSTS} hosts"}
                lane = self._add_lane(remote)
                hosts_mod.save_hosts(self._remotes(), self._hosts_path)
                if lane.tunnel is not None:
                    lane.tunnel.ensure()
        elif action == "remove":
            name = body.get("name")
            if name == hosts_mod.LOCAL:
                return 400, {"error": "cannot_remove_local", "message": "this machine is always a host"}
            with self._lanes_lock:
                removed = self._lanes.pop(name, None) if isinstance(name, str) else None
                if removed is not None:
                    self._retired.append(removed.store)
                    del self._retired[:-hosts_mod.MAX_HOSTS]
            if removed is None:
                return 400, {"error": "host_unknown", "message": f"no host named {name}"}
            removed.store.close()
            if removed.tunnel is not None:
                removed.tunnel.close()
            hosts_mod.save_hosts(self._remotes(), self._hosts_path)
        else:
            return 400, {"error": "bad_request", "message": 'action must be "add" or "remove"'}
        return 200, {"ok": True, "hosts": [self._host_row(lane) for lane in self._lane_list()]}

    def _route(self, request: dict) -> tuple[Lane | None, tuple]:
        """Which machine answers. A named host wins; otherwise the least-busy host
        that has the named model, this machine winning a tie; a question naming no
        model goes to this machine, whose pick `/status` shows. Nothing suitable →
        this machine, where the job fails the way it always did (`model_not_installed`)."""
        lanes = self._lane_list()
        local = lanes[0]
        wanted, model = request.get("host"), request.get("model")
        if wanted is not None:
            lane = next((lane for lane in lanes if lane.name == wanted), None)
            if lane is None:
                return None, (400, {"error": "host_unknown", "message": f"no host named {wanted}"})
            found = self._installed_for(lane)
            if found.models is None:
                return None, (409, {"error": "host_unreachable",
                                    "message": found.error or f"{wanted} is not answering"})
            if model and model not in _names(found.models):
                return None, (409, {"error": "model_not_on_host", "message": f"{model} is not installed on {wanted}"})
            return lane, ()
        if len(lanes) == 1 or not model:
            return local, ()
        candidates = [lane for lane in lanes
                      if (found := self._installed_for(lane)).models is not None and model in _names(found.models)]
        if not candidates:
            return local, ()
        return min(candidates, key=lambda lane: (lane.store.load(), not lane.is_local)), ()

    def _find_job(self, job_id: str) -> dict | None:
        with self._lanes_lock:
            stores = [lane.store for lane in self._lanes.values()] + self._retired
        for store in [*stores, self._agent_jobs]:
            job = store.get(job_id)
            if job is not None:
                return job
        return None

    # -- internals ------------------------------------------------------------
    def _installed_for(self, lane: Lane) -> Installed:
        return self._installed() if lane.is_local else self._installed_remote(lane)

    def _installed_remote(self, lane: Lane) -> Installed:
        """Another machine's models, read through its tunnel and cached like `_installed`.
        A tunnel that is not up yet is reported, not waited for: this runs on a request thread."""
        assert lane.remote is not None and lane.tunnel is not None
        state, port, tunnel_error = lane.tunnel.ensure()
        runtime = runtimes_mod.on_host(lane.remote.kind, port, lane.name)
        if port is None:
            detail = TUNNEL_STARTING_MESSAGE if state == STARTING else (tunnel_error or "the tunnel is down")
            return Installed(runtime, None, None, f"host_unreachable: {detail}")
        now = time.monotonic()
        with lane.lock:
            stamp, cached_port, cached = lane.cache
            if cached is not None and cached.models is not None and cached_port == port \
                    and now - stamp < MODELS_TTL_SECONDS:
                return cached
        try:
            found = Installed(runtime, runtime.list_models(self._opener), runtime.version(self._opener), None)
        except RuntimeDown as err:
            found = Installed(runtime, None, None, f"runtime_down: {runtime.describe()}: {err}")
        except RuntimeRefused as err:
            found = Installed(runtime, None, None, f"runtime_error: {err.message}")
        with lane.lock:
            lane.cache = (now, port, found)
        return found

    def _installed(self) -> Installed:
        """The first server that answers, cached for MODELS_TTL_SECONDS; a miss
        (`models` None) is never cached, so the next request probes again.

        A server that answers with an error (a web app squatting on a probed
        port, an endpoint that is not a model list) is passed over like one
        that is down; the error is reported only when it is the ONE server the
        user configured, since then it is the answer.
        """
        now = time.monotonic()
        with self._models_lock:
            stamp, cached = self._models_cache
            if cached is not None and cached.models is not None and now - stamp < MODELS_TTL_SECONDS:
                return cached
        candidates = runtimes_mod.candidates()
        found: Installed | None = None
        problems = []
        for candidate in candidates:
            try:
                found = Installed(candidate, candidate.list_models(self._opener), candidate.version(self._opener), None)
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
            found = Installed(candidates[0], None, None, error)
        with self._models_lock:
            self._models_cache = (now, found)
        return found

    def _host_row(self, lane: Lane, found: Installed | None = None) -> dict:
        found = found or self._installed_for(lane)
        counts = lane.store.counts()
        tok_s, prefill_tok_s = lane.speed.rates()
        if lane.tunnel is None:
            tunnel = NONE
        else:
            tunnel, _, _ = lane.tunnel.snapshot()
        return {
            "name": lane.name,
            "local": lane.is_local,
            "kind": found.runtime.kind,
            "reachable": found.models is not None,
            "models": [str(m["name"]) for m in found.models or [] if models_mod.is_chat_model(m)],
            "queue": counts["queued"] + counts["running"],
            "running": counts["running"] > 0,
            "tok_s": tok_s,
            "prefill_tok_s": prefill_tok_s,
            "tunnel": tunnel,
            "error": found.error,
        }

    def _status(self) -> dict:
        lanes = self._lane_list()
        found_by_lane = [(lane, self._installed_for(lane)) for lane in lanes]
        runtime, models, runtime_version, error = found_by_lane[0][1]
        pinned = os.environ.get(PIN_ENV) or None
        picked = models_mod.pick_model(models or [], pinned)
        union: list[dict] = []
        model_hosts: dict[str, list[str]] = {}
        for lane, found in found_by_lane:
            for model in found.models or []:
                if not models_mod.is_chat_model(model):
                    continue
                name = str(model["name"])
                if name not in model_hosts:
                    union.append(model)
                model_hosts.setdefault(name, []).append(lane.name)
        totals = {"queued": 0, "running": 0}
        for lane in lanes:
            for key, value in lane.store.counts().items():
                totals[key] += value
        return {
            "version": STATUS_VERSION,
            "plugin": self.name,
            "runtime": runtime.kind,
            "runtime_label": runtime.label,
            "runtime_url": runtime.base,
            "reachable": models is not None,
            "runtime_version": runtime_version,
            "error": error,
            "models": union,
            "picked_model": picked,
            "pinned_model": pinned,
            "preference": list(models_mod.PREFERENCE),
            **totals,
            "hosts": [self._host_row(lane, found) for lane, found in found_by_lane],
            "model_hosts": model_hosts,
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

    def _run_chat_on(self, lane_name: str, request: dict) -> dict:
        with self._lanes_lock:
            lane = self._lanes.get(lane_name)
        if lane is None:
            raise jobs_mod.JobFailure("host_unknown", f"{lane_name} was removed before this question ran")
        if lane.tunnel is not None and lane.tunnel.wait_up() is None:
            _, _, tunnel_error = lane.tunnel.snapshot()
            raise jobs_mod.JobFailure("host_unreachable", f"{lane.name}: {tunnel_error or 'the tunnel did not open'}")
        runtime, models, _, error = self._installed_for(lane)
        if models is None:
            code = "host_unreachable" if not lane.is_local else "runtime_down"
            raise jobs_mod.JobFailure(code, error or f"{runtime.label} is not answering")
        # The pin names a model on THIS machine; another host picks from its own list.
        pinned = (os.environ.get(PIN_ENV) or None) if lane.is_local else None
        model, problem = models_mod.resolve_requested(request.get("model"), models, pinned)
        if problem == "model_not_installed":
            if not lane.is_local:
                raise jobs_mod.JobFailure("model_not_on_host",
                                          f"{request.get('model')} is not installed on {lane.name}")
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
            code = "runtime_down" if lane.is_local else "host_unreachable"
            raise jobs_mod.JobFailure(code, f"{runtime.label} stopped answering: {err}") from err
        except RuntimeRefused as err:
            raise jobs_mod.JobFailure("runtime_error", err.message) from err
        timings = _timings(result)
        lane.speed.record(timings)
        return {**result, "timings": timings}


def _ssh_tunnel(remote: hosts_mod.RemoteHost) -> SshTunnel:
    return SshTunnel(remote.ssh, remote.port)


def _names(models: list[dict]) -> set[str]:
    return {str(m.get("name", "")) for m in models}


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
                {"function": {"name": str(c.get("name", "")), "arguments": typed(c.get("arguments"), dict, {})}}
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


def _validate_agent(raw: object, messages: list[dict], tools: list[dict] | None, model: str | None,
                    agent_runtimes: Mapping[str, AgentRuntime]) -> tuple[dict | None, str | None, str | None]:
    """An agent turn: `{runtime, …}` naming a registered runtime, one system +
    one user message and no tools. Returns (agent, model, problem): the model is
    the body's or the runtime's default, and the runtime has checked it, as it
    checks its own fields (`agents.py`)."""
    if not isinstance(raw, dict):
        return None, None, "agent must be an object {runtime, …}"
    runtime_name = raw.get("runtime")
    runtime = agent_runtimes.get(runtime_name) if isinstance(runtime_name, str) else None
    if runtime is None:
        return None, None, f"agent.runtime must be one of {tuple(agent_runtimes)}"
    agent, problem = runtime.validate_agent(raw)
    if problem:
        return None, None, problem
    if tools or [m["role"] for m in messages] != ["system", "user"]:
        return None, None, "an agent takes one system and one user message, and no tools"
    model = model if model is not None else runtime.default_model
    problem = runtime.model_error(model)
    if problem:
        return None, None, problem
    return {"runtime": runtime.name, **(agent or {})}, model, None


def _validate_chat(body: dict | None,
                   agent_runtimes: Mapping[str, AgentRuntime] | None = None) -> tuple[dict, str | None]:
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

    raw_model = body.get("model")
    if raw_model is not None and (not isinstance(raw_model, str) or not raw_model.strip() or len(raw_model) > 128):
        return {}, "model must be a non-empty string"
    model: str | None = raw_model.strip() if isinstance(raw_model, str) else None

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
    if think is not None and not isinstance(think, bool) and think not in THINK_LEVELS:
        return {}, f"think must be a boolean or one of {', '.join(THINK_LEVELS)}"

    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED):
        return {}, f"seed must be an integer from 0 to {MAX_SEED}"

    host = body.get("host")
    if host is not None and (not isinstance(host, str) or not (host == hosts_mod.LOCAL
                                                               or hosts_mod.alias_problem(host) is None)):
        return {}, "host must be a host name from /status"

    agent = None
    if body.get("agent") is not None:
        agent, model, problem = _validate_agent(body.get("agent"), messages, tools, model, agent_runtimes or {})
        if problem:
            return {}, problem

    return {
        "agent": agent,
        "messages": messages,
        "tools": tools,
        "num_ctx": num_ctx,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "format": response_format,
        "think": think,
        "seed": seed,
        "host": host,
    }, None
