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
voidcraft-world-bridge 0.1.1 — listening on http://127.0.0.1:7682
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
- **No shell, no files.** It routes chat requests to your model server (and, if you add them,
  your other machines' model servers over SSH) and nothing else.
- **Your prompts are not kept.** A request is dropped from memory once it is answered.
- **Stdlib only.** Zero third-party dependencies.

## The local-llm API

Mounted at `/plugin/local-llm/`:

| Route | Behaviour |
|---|---|
| `GET /status` | Which server answered (`runtime`, `runtime_label`, `runtime_url`), whether it is `reachable`, the installed **chat** models, and `picked_model`. Never an error when no server runs; `reachable: false` is a normal state. `hosts` lists every machine (below) and `model_hosts` which ones have each model. |
| `POST /chat` | `{system, user}` or `{messages, tools?}`, plus `model?, host?, max_tokens?, temperature?, format?, think?, num_ctx?, seed?` → **202** `{job_id, host}` at once. `400` on a bad body, `429` when 4 requests already wait for that machine; with `host`, `400 host_unknown`, `409 host_unreachable` or `409 model_not_on_host`. |
| `GET /job?id=` | `queued` → `running` → `done` (`result.text`, `result.tool_calls`, token counts, `total_ms`) or `failed` (`error.code`: `runtime_down` · `runtime_error` · `no_model` · `model_not_installed` · `host_unreachable` · `model_not_on_host`). Every job names the `host` that answered and its `timings`. |
| `POST /hosts` | `{action: "add", ssh, name?, kind?, port?}` or `{action: "remove", name}` → `{ok, hosts}`. Answers on this machine's loopback only. |

Generation is a job, not a request: one answer can take a minute, and one model generates one
answer at a time — two at once would only split the same memory bandwidth.

## Compute hosts

Another machine on your desk can answer questions too: each one is its own queue, so two
questions decode at once instead of in turn. The bridge reaches it over **SSH**, so its model
server stays bound to its own loopback and nothing listens on your network.

1. On the other machine, run a model server bound to `127.0.0.1` (Ollama does by default).
2. On this machine, give it a `Host` entry in `~/.ssh/config` and connect once by hand, so its
   key and host key are set up: `ssh studio`.
3. Add it: `POST /plugin/local-llm/hosts` with `{"action": "add", "ssh": "studio"}`, or list it
   in `~/.config/voidcraft-world-bridge/hosts.json` (or the file `VOIDCRAFT_LOCAL_LLM_HOSTS`
   names):

```json
{ "version": 1, "hosts": [{ "name": "studio", "ssh": "studio", "kind": "ollama", "port": 11434 }] }
```

- **Only an alias.** A host must be a literal `Host` line in your SSH config, and the bridge
  never stores an address, user or key. `ssh` runs with `BatchMode=yes`, so a host that would
  prompt for anything fails instead of hanging.
- **Routing.** A question naming `host` goes there. One naming only a `model` goes to the
  least-busy machine that has it, this one winning a tie. One naming neither stays here.
- **Awake while used.** The tunnel's SSH session holds the other machine awake (`caffeinate`
  on macOS, `systemd-inhibit` on Linux) and closes after 30 idle minutes, so it can sleep again.
- **Speed.** `/status` reports each host's recent `tok_s` and `prefill_tok_s`, so a caller can
  tell a fast machine from a slow one before it waits on it.

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
page. POST bodies are capped at 4 KB unless the plugin sets `max_body_bytes`. A plugin that
fails to load is skipped, and a handler that raises becomes a `500`; handler failures are isolated. Plugins are trusted Python code in the same process;
this is not a sandbox. Full contract: `voidcraft_world_bridge/plugins.py`.

A complete, runnable one to copy:
[`examples/hello-plugin`](https://github.com/voidcraft-world/voidcraft-world-bridge/tree/main/examples/hello-plugin)
— one file, every part of the contract, loaded and exercised by the test suite so it cannot
drift from it.

```bash
VOIDCRAFT_BRIDGE_PLUGINS=$PWD/examples/hello-plugin voidcraft-world-bridge --port 7791
curl 'http://127.0.0.1:7791/plugin/hello/greeting?name=you'
```

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

## Optional plugin SDK

Existing plugins continue to work. New plugins can import these stdlib-only public modules:

| Module | Purpose |
| --- | --- |
| `policy` | `CallerPolicy`: optional stricter origins, local hosts and proxy/no-cors restrictions; assign it to `plugin.request_policy` |
| `routing` | `Route(method, path, handler)` and `RoutedPlugin(name, routes)` generate dispatch and discovery from one table |
| `services` | `ManagedService`: start a loopback child, probe readiness, back off after failure, stop after idle time, and close owned resources |
| `jobs` | `BackgroundJob` rejects overlapping work; `JobStore` queues work with configurable pending/retained limits |
| `testing` | `PluginClient` validates and exercises a plugin without sockets |

Route handlers take `(query, body)` and return the existing response tuples. `operations` in
`/snapshot.plugins[name]` adds method/path/description metadata; `routes` keeps its old shape.
Caller policies can only narrow the host guards. They are not authentication of local processes.

A plugin owning resources should implement `close()`. `make_server()` calls it on server close;
the loader also closes a factory result that fails validation.
Embedding hosts constructing their own server must call `close_plugins()` themselves. A background
callable cannot be forcibly cancelled: both job helpers reject new submissions on close, and accepted
work drains while the process remains alive. Detached durable jobs stay outside these helpers.

`ManagedService` accepts trusted command/environment callbacks, a log path, and a readiness predicate.
The child must bind loopback and verify the supplied per-spawn key. No command comes from an HTTP body.
Call its `close()` from your plugin's `close()`; request bodies and credentials are never logged by it.
Requests use absolute paths, bypass system proxies and do not follow redirects. A stopped or failed
child cannot receive new requests; probes from an older launch cannot contact its replacement.

Check a trusted plugin before starting the server:

```bash
voidcraft-world-bridge check-plugin ./examples/hello-plugin
```

This imports the plugin and calls its factory and `routes()`, then closes it; it is not a sandbox.
Startup failures are reported in `plugin_diagnostics` on `/snapshot` without absolute paths or exception
messages. Diagnostic details stay in the local startup log. The structural checker does not invoke
application handlers or test application semantics; use `PluginClient` for those tests.
