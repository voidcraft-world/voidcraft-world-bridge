"""The Ollama adapter — the one module that knows Ollama's native wire format.

`openai_compat.py` is its sibling for every OpenAI-compatible server (LM Studio,
llama.cpp, vLLM, …). Both expose the same `list_models` / `version` / `chat`, each
taking the server's base URL; `plugin.py` never learns which one answered.

── TIMEOUTS ──────────────────────────────────────────────────────────────────
Two very different budgets. `/api/tags` and `/api/version` are loopback JSON:
if they take more than a couple of seconds Ollama is not slow, it is absent,
and `/status` runs on a request thread. `/api/chat` runs on the plugin's own
worker thread and may first have to page a 23 GB model in from disk, so its
budget is minutes — for the whole answer, not per line (`transport.py`).

── STREAMED ──────────────────────────────────────────────────────────────────
`/api/chat` is always asked to stream: one JSON line per token, then a final
line (`"done": true`) carrying the counters a one-shot reply would have. The
lines are how a running job shows the answer being written (`progress.py`).
"""
from __future__ import annotations

import os

from voidcraft_world_bridge.local_llm.jobs import Report
from voidcraft_world_bridge.local_llm.progress import ChatProgress
from voidcraft_world_bridge.local_llm.transport import Opener, RuntimeDown, RuntimeRefused, http_json, http_ndjson
from voidcraft_world_bridge.shape import typed

__all__ = ["RuntimeDown", "RuntimeRefused", "base_url", "chat", "list_models", "version"]

KIND = "ollama"
LABEL = "Ollama"

URL_ENV = "VOIDCRAFT_OLLAMA_URL"
DEFAULT_URL = "http://127.0.0.1:11434"
"""NOT Ollama's own OLLAMA_HOST, which is a BIND address (`0.0.0.0:11434`) and
not a URL. Reusing it would turn a user's server config into a broken client
URL."""

META_TIMEOUT_SECONDS = 2.0
CHAT_TIMEOUT_SECONDS = 300.0

DEFAULT_NUM_CTX = 16384
"""Set explicitly on every chat. Ollama's default context has changed between
releases and has been as small as 2–4K, which silently truncates a question's
sources from the FRONT — the model then answers without ever seeing them."""

KEEP_ALIVE = "10m"
"""How long the model stays loaded after a request. A 23 GB model is worth
keeping warm between the questions of one session, and worth unloading after:
it holds most of a 36 GB Mac's GPU budget while resident."""


def base_url() -> str:
    return os.environ.get(URL_ENV, DEFAULT_URL).rstrip("/")


def missing_model_hint(model: str) -> str:
    return f"{model} is not installed — run `ollama pull {model}`"


NO_MODEL_HINT = "No chat model installed — run `ollama pull qwen3.5:9b`"


def list_models(base: str, opener: Opener | None = None) -> list[dict]:
    """Installed models, each trimmed to what `/status` shows."""
    payload = http_json(f"{base}/api/tags", None, META_TIMEOUT_SECONDS, opener)
    models = []
    for entry in payload.get("models") or []:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        details = typed(entry.get("details"), dict, {})
        models.append({
            "name": str(entry["name"]),
            "digest": str(entry["digest"]) if entry.get("digest") else None,
            "size_bytes": int(entry.get("size") or 0),
            "details": {
                "family": details.get("family"),
                "families": details.get("families") or [],
                "parameter_size": details.get("parameter_size"),
                "quantization_level": details.get("quantization_level"),
            },
        })
    return models


def version(base: str, opener: Opener | None = None) -> str | None:
    return http_json(f"{base}/api/version", None, META_TIMEOUT_SECONDS, opener).get("version")


def chat(base: str, model: str, messages: list[dict], *, max_tokens: int, temperature: float,
         response_format: dict | str | None, think: bool | str | None,
         tools: list[dict] | None = None, num_ctx: int | None = None,
         seed: int | None = None, opener: Opener | None = None,
         on_progress: Report | None = None) -> dict:
    """One chat turn. Returns the text plus the counters Ollama reports.

    `seed` is sent only when the caller set it; without one Ollama samples freshly
    every call, which is what a one-off question wants.

    `think` is sent only when the caller set it. A model without a thinking mode
    rejects the field outright on some Ollama builds ("does not support
    thinking"), so on exactly that error the turn is re-sent once without it —
    the caller asked for "don't think", and a model that cannot think already
    satisfies that. A level (`"low"`) is passed as given, for a model that reads
    levels only (gpt-oss ignores `false`). The rejection comes before the first
    token — as an HTTP error, or as an error line in its place — so a retry is
    only ever made while nothing has been written.

    `on_progress` receives the answer so far as it is written (`progress.py`).
    The turn is streamed whether or not anyone is watching: one code path, and
    streaming is what lets the whole answer keep a deadline. The result is the
    same either way — the text and tool calls are the lines' deltas joined, and
    the counters come from the final line, as they came from the one-shot reply.
    """
    body: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {"num_ctx": num_ctx or DEFAULT_NUM_CTX, "temperature": temperature, "num_predict": max_tokens},
    }
    if response_format is not None:
        body["format"] = response_format
    if tools:
        body["tools"] = tools
    if think is not None:
        body["think"] = think
    if seed is not None:
        body["options"]["seed"] = seed

    progress = ChatProgress(on_progress)
    progress.begin()
    try:
        final, calls = _stream(base, body, progress, opener)
    except RuntimeRefused as err:
        if "think" in body and "think" in err.message.lower() and not progress.writing:
            body.pop("think")
            final, calls = _stream(base, body, progress, opener)
        else:
            raise
    progress.finish()

    return {
        "text": progress.text,
        "tool_calls": _tool_calls(calls),
        "model": str(final.get("model") or model),
        "prompt_tokens": int(final.get("prompt_eval_count") or 0),
        "output_tokens": int(final.get("eval_count") or 0),
        "total_ms": _ns_to_ms(final.get("total_duration")),
        "load_ms": _ns_to_ms(final.get("load_duration")),
        "prompt_ms": _ns_to_ms(final.get("prompt_eval_duration")),
        "output_ms": _ns_to_ms(final.get("eval_duration")),
    }


def _stream(base: str, body: dict, progress: ChatProgress, opener: Opener | None) -> tuple[dict, list]:
    """Read one streamed reply into `progress`. Returns Ollama's final line (the
    counters) and the raw tool calls, which arrive whole, each in its own line."""
    calls: list = []
    with http_ndjson(f"{base}/api/chat", body, CHAT_TIMEOUT_SECONDS, opener) as lines:
        for line in lines:
            message = typed(line.get("message"), dict, {})
            new_calls = typed(message.get("tool_calls"), list, [])
            calls.extend(new_calls)
            progress.add(typed(message.get("content"), str, ""), typed(message.get("thinking"), str, ""),
                         tool_calls=bool(new_calls))
            if line.get("done") is True:
                return line, calls
    raise RuntimeDown("the reply ended before Ollama finished it")


def _ns_to_ms(value: object) -> int | None:
    return int(value) // 1_000_000 if isinstance(value, (int, float)) and value >= 0 else None


def _tool_calls(raw: object) -> list[dict]:
    """The model's tool calls as `[{name, arguments}]` — Ollama's shape, flattened and typed."""
    calls = []
    for call in raw if isinstance(raw, list) else []:
        fn = call.get("function") if isinstance(call, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        args = fn.get("arguments")
        calls.append({"name": fn["name"], "arguments": typed(args, dict, {})})
    return calls
