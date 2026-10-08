"""The OpenAI-compatible adapter: LM Studio, llama.cpp's `llama-server`, vLLM, Jan, …

Every one of them speaks `GET /v1/models` and `POST /v1/chat/completions`, so one
adapter covers them all. It exposes the same `list_models` / `version` / `chat`
as `ollama.py`, each taking the server's base URL (the one ending in `/v1`), and
answers in the same result shape, so `plugin.py` never learns which one answered.

── WHAT DOES NOT TRANSLATE, AND WHAT HAPPENS INSTEAD ─────────────────────────
- `num_ctx`: the context size is fixed when the SERVER loads the model, so it is
  not sent. Load the model with enough context for your prompts.
- `think`: there is no portable field for it. Not sent; a reply that still
  starts with a `<think>…</think>` block has the block removed, because the
  caller asked for an answer, not the reasoning.
- `format`: sent as `response_format` with `type: "json_schema"` — the one form
  all of these servers accept. A bare `"json"` becomes the schema `{"type":
  "object"}`.
- Tool calls: the conversation is held in Ollama's shape (arguments as an
  object, a tool reply naming its tool). Here a call carries an `id` and its
  arguments as a JSON string, and a tool reply names that `id`. The ids are
  minted per request and matched to replies by tool name, in order.
"""
from __future__ import annotations

import json
import os
import re
import time

from voidcraft_world_bridge.local_llm.transport import Opener, RuntimeRefused, http_json

KIND = "openai"
LABEL = "OpenAI-compatible server"

URL_ENV = "VOIDCRAFT_LOCAL_LLM_URL"
"""Point the bridge at one server explicitly, e.g. `http://127.0.0.1:1234/v1`.
Set, it is the only server asked; unset, the well-known local ports are probed
(`runtimes.py`)."""

META_TIMEOUT_SECONDS = 2.0
CHAT_TIMEOUT_SECONDS = 300.0

_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def configured_url() -> str | None:
    raw = os.environ.get(URL_ENV, "").strip()
    return raw.rstrip("/") or None


def missing_model_hint(model: str) -> str:
    return f"{model} is not available on this server — load it there first"


NO_MODEL_HINT = ("No chat model loaded — load one in LM Studio, or start llama-server "
                 "with -m <model.gguf>")


def list_models(base: str, opener: Opener | None = None) -> list[dict]:
    """The server's models, in the same trimmed shape `ollama.list_models` gives.

    A server that answers without a `data` list is not one of these — some other
    program on a well-known port — and is refused rather than read as "no models".
    """
    payload = http_json(f"{base}/models", None, META_TIMEOUT_SECONDS, opener)
    entries = payload.get("data")
    if not isinstance(entries, list):
        raise RuntimeRefused(200, "not an OpenAI-compatible server (no model list)")
    models = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
            continue
        models.append({
            "name": entry["id"],
            "size_bytes": 0,
            "details": {"family": None, "families": [], "parameter_size": None, "quantization_level": None},
        })
    return models


def version(base: str, opener: Opener | None = None) -> str | None:  # noqa: ARG001
    """There is no standard version route; the server's name is what `/status` shows."""
    return None


def chat(base: str, model: str, messages: list[dict], *, max_tokens: int, temperature: float,
         response_format: dict | str | None, think: bool | None,  # noqa: ARG001 — see the module note
         tools: list[dict] | None = None, num_ctx: int | None = None,  # noqa: ARG001 — see the module note
         seed: int | None = None, opener: Opener | None = None) -> dict:
    """One non-streaming chat turn, answered in the same shape as `ollama.chat`."""
    body: dict = {
        "model": model,
        "messages": to_openai_messages(messages),
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format is not None:
        schema = response_format if isinstance(response_format, dict) else {"type": "object"}
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
    if tools:
        body["tools"] = tools
    if seed is not None:
        body["seed"] = seed

    started = time.monotonic()
    payload = http_json(f"{base}/chat/completions", body, CHAT_TIMEOUT_SECONDS, opener)
    elapsed_ms = int((time.monotonic() - started) * 1000)

    choices = payload.get("choices") if isinstance(payload.get("choices"), list) else []
    first = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    content = message.get("content") if isinstance(message.get("content"), str) else ""
    return {
        "text": _THINK_BLOCK.sub("", content, count=1),
        "tool_calls": _tool_calls(message.get("tool_calls")),
        "model": str(payload.get("model") or model),
        "prompt_tokens": _count(usage.get("prompt_tokens")),
        "output_tokens": _count(usage.get("completion_tokens")),
        "total_ms": elapsed_ms,
        "load_ms": None,
    }


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """The plugin's conversation (Ollama's shape) → the OpenAI shape. See the module note."""
    out: list[dict] = []
    pending: list[tuple[str, str]] = []  # (call id, tool name) not yet answered, in call order
    for index, turn in enumerate(messages):
        role = turn["role"]
        if role == "assistant" and turn.get("tool_calls"):
            calls = []
            for position, call in enumerate(turn["tool_calls"]):
                function = call.get("function") or {}
                call_id = f"call_{index}_{position}"
                name = str(function.get("name", ""))
                calls.append({"id": call_id, "type": "function", "function": {
                    "name": name, "arguments": json.dumps(function.get("arguments") or {}),
                }})
                pending.append((call_id, name))
            out.append({"role": "assistant", "content": turn.get("content") or "", "tool_calls": calls})
        elif role == "tool":
            name = turn.get("tool_name")
            match = next((entry for entry in pending if entry[1] == name), pending[0] if pending else None)
            if match is not None:
                pending.remove(match)
            out.append({"role": "tool", "tool_call_id": match[0] if match else f"call_{index}",
                        "content": turn["content"]})
        else:
            out.append({"role": role, "content": turn["content"]})
    return out


def _tool_calls(raw: object) -> list[dict]:
    """The model's tool calls as `[{name, arguments}]` — arguments parsed from their JSON string."""
    calls = []
    for call in raw if isinstance(raw, list) else []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except ValueError:
                arguments = {}
        calls.append({"name": function["name"], "arguments": arguments if isinstance(arguments, dict) else {}})
    return calls


def _count(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) and value >= 0 else 0
