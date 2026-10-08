# voidcraft-world-bridge

[![PyPI](https://img.shields.io/pypi/v/voidcraft-world-bridge)](https://pypi.org/project/voidcraft-world-bridge/)
[![CI](https://github.com/voidcraft-world/voidcraft-world-bridge/actions/workflows/ci.yml/badge.svg)](https://github.com/voidcraft-world/voidcraft-world-bridge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/voidcraft-world-bridge)](https://pypi.org/project/voidcraft-world-bridge/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A small loopback server that lets [voidcraft.world](https://voidcraft.world) talk to
**your own local LLM**, so it can answer questions about your worlds or command a side in
Arena World.

```bash
uvx voidcraft-world-bridge
```

That's the whole install. It finds your model server on its own and listens on
`http://127.0.0.1:7682` until you press Ctrl-C:

```
voidcraft-world-bridge 0.1.0 — listening on http://127.0.0.1:7682
  plugin: local-llm  → http://127.0.0.1:7682/plugin/local-llm/…
  local model:  Ollama at http://127.0.0.1:11434 → qwen3.6:35b-a3b
  Ctrl-C to stop
```

## Your model server

Run any one of these and the bridge finds it, asking in this order:

| Server | Where the bridge looks |
|---|---|
| [Ollama](https://ollama.com) | `http://127.0.0.1:11434` |
| [LM Studio](https://lmstudio.ai) (start its local server) | `http://127.0.0.1:1234/v1` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` | `http://127.0.0.1:8080/v1` |

Anything else that speaks the OpenAI API (vLLM, Jan, a custom port) works too. Point the
bridge at it, and it asks only that server:

```bash
VOIDCRAFT_LOCAL_LLM_URL=http://127.0.0.1:8000/v1 uvx voidcraft-world-bridge
```

**Which model answers:** the strongest chat model you have, by a built-in preference list
(Qwen 3.6 35B-A3B first). Embedding models are never picked. Pin one with
`VOIDCRAFT_LOCAL_LLM_MODEL=<name>`. The bridge never downloads a model; install one with your
server first (`ollama pull qwen3.5:9b` is a good small start).

**What the OpenAI-compatible path cannot do:** set the context size per request. Load your model
with enough context (16K or more) in the server itself.

## Why it exists

A web page cannot call `localhost` services directly: CORS and Chrome's Local Network
Access stop it, on purpose. The bridge is the one door through, and it is narrow:

- **Loopback only.** It binds `127.0.0.1`. There is no flag to bind anything wider.
- **Origin allowlist.** Only `voidcraft.world` and pages served from your own machine
  may call it. Any other site gets a `403`.
- **DNS-rebinding guard.** A request addressed to any hostname other than `localhost` /
  `127.0.0.1` / `[::1]` gets a `403`, even when the origin looks right.
- **No shell, no files.** It routes chat requests to your model server and nothing else.
- **Your prompts are not kept.** A request is dropped from memory once it is answered.
- **Stdlib only.** Zero third-party dependencies.

## The local-llm API

Mounted at `/plugin/local-llm/`:

| Route | Behaviour |
|---|---|
| `GET /status` | Which server answered (`runtime`, `runtime_label`, `runtime_url`), whether it is `reachable`, the installed **chat** models, and `picked_model`. Never an error when no server runs; `reachable: false` is a normal state. |
| `POST /chat` | `{system, user}` or `{messages, tools?}`, plus `model?, max_tokens?, temperature?, format?, think?, num_ctx?, seed?` → **202** `{job_id}` at once. `400` on a bad body, `429` when 4 requests already wait. |
| `GET /job?id=` | `queued` → `running` → `done` (`result.text`, `result.tool_calls`, token counts, `total_ms`) or `failed` (`error.code`: `runtime_down` · `runtime_error` · `no_model` · `model_not_installed`). |

Generation is a job, not a request: one answer can take a minute, and one model generates one
answer at a time — two at once would only split the same memory bandwidth.

## Commands

```bash
voidcraft-world-bridge              # serve (foreground)
voidcraft-world-bridge status       # is one running? what does it mount?
voidcraft-world-bridge --version
voidcraft-world-bridge --port 7700  # or VOIDCRAFT_BRIDGE_PORT=7700
```

## Plugins

Everything the bridge can do is a plugin mounted at `/plugin/<name>/…`. `local-llm` is built
in; add your own by pointing the bridge at a directory containing a `bridge_plugin.py`:

```bash
VOIDCRAFT_BRIDGE_PLUGINS=/path/to/my-plugin voidcraft-world-bridge
```

or list directories in `~/.config/voidcraft-world-bridge/plugins.json`:

```json
{ "version": 1, "plugins": ["/path/to/my-plugin"] }
```

A plugin module defines `PLUGIN_NAME` and `create_plugin(context)`, returning an object
with `routes()`, `handle_get(subpath, query)` and `handle_post(subpath, query, body)`.
Handlers return `(status, dict)` for JSON or `(status, bytes, content_type)` for a raw
page. A plugin that fails to load is skipped, and a handler that raises becomes a `500`;
a plugin can never take the bridge down. Full contract: `voidcraft_world_bridge/plugins.py`.

## Development

```bash
uv sync
uv run pytest                                # no model server needed; every test fakes one
uv run voidcraft-world-bridge --port 7791    # a spare port, if a bridge already holds 7682
```

[CONTRIBUTING.md](CONTRIBUTING.md) has the two rules every change must keep (stdlib only, nothing
private) and the checks CI runs. Found something exploitable? [SECURITY.md](SECURITY.md) says how
to report it privately.

## Remote access (optional)

To reach the bridge through `tailscale serve`, name the machine explicitly:

```bash
VOIDCRAFT_BRIDGE_ALLOWED_HOSTS=mymac.tailXXXX.ts.net voidcraft-world-bridge
```

Never use `tailscale funnel` or a public tunnel: that publishes your bridge to the internet.
