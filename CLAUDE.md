# voidcraft-world-bridge — Agent Guide

A stdlib-only loopback HTTP server that lets a page on voidcraft.world reach this machine
through plugins, behind a Host + Origin allowlist. Its built-in `local-llm` plugin is the
reason people run it: `uvx voidcraft-world-bridge`, and voidcraft.world can use the model
server they already run (Ollama, LM Studio, llama.cpp, any OpenAI-compatible one).

`AGENTS.md` is a symlink to this file, so every coding agent reads the same guide.

## The two rules (both are tests)

1. **Stdlib only.** `tests/test_purity.py` walks every import. A dependency is a
   `pyproject.toml` decision, never an import that slips in. Reason: this code runs on a
   stranger's machine with loopback access to their model, and every dependency is one
   more thing they must trust.
2. **Nothing private.** The same test scans every file in the repo (`.py`, `.md`, `.toml`)
   for the shapes a leak takes — `FORBIDDEN` there is the list; do not restate it. A host
   that embeds this package keeps its own notes on its own side.

## Layout

| Module | Owns |
|---|---|
| `guards.py` | `host_allowed` (DNS rebinding), `origin_allowed`, `is_no_cors_browser_request`, the origin allowlist. The security reasoning lives in the docstrings here. |
| `plugins.py` | Plugin discovery, loading, dispatch, body caps, the `/snapshot` advertisement. `discover_plugin_dirs(env, config_path, env_var=)` lets a host keep its own names. |
| `server.py` | `BridgeHandler` (guards → plugin → `/snapshot` → host routes → 404), `bridge_block()`, `make_server` (loopback bind, no wider option). |
| `cli.py` | `voidcraft-world-bridge [status] [--port] [--version]`, argparse; the banner names the model server `local-llm` found. |
| `builtins.py` | The plugins every bridge mounts with no config (`local-llm`), and the one door for a host's agent runtimes. |
| `local_llm/plugin.py` | `/status`, `/chat` (202 + job), `/job`; body validation (it arrives from a browser). |
| `local_llm/runtimes.py` | Which model server answers: probe Ollama → LM Studio → llama.cpp, or the one configured URL. |
| `local_llm/ollama.py`, `openai_compat.py` | The two wire formats, same three functions; what does not translate to OpenAI's shape is in the latter's docstring. |
| `local_llm/models.py` | Which installed model answers; reads both Ollama tags and OpenAI-style ids. |
| `local_llm/jobs.py` | The bounded single-worker queue, and why generation is a job. |
| `local_llm/agents.py` | The agent-runtime contract. The package ships NONE; a host registers its own. |

## Extending it as a host

Subclass `BridgeHandler`, name routes in `GET_ROUTES` / `POST_ROUTES` (path → zero-arg
method name), override `snapshot_payload()` keeping `bridge_block()` under `bridge` and
the `plugins` key. The guards run in `do_GET`/`do_POST` ahead of every route, so a host
route cannot forget them (`tests/test_server.py` asserts that). Mount the built-ins with
`load_plugins(dirs, context, preloaded=builtin_plugins(context, agent_runtimes=[…]))`.

## Contracts clients depend on

- `GET /snapshot` → `{bridge: {name, version, protocol}, plugins?: {name: {routes}}}`.
  **Additive-only**: a host may add keys, never change these. `protocol` bumps only when a
  key changes meaning. voidcraft.world reads `plugins` to decide what it can offer.
- `GET /plugin/local-llm/status` — additive-only as well (`STATUS_VERSION`).
- Port `7682` by default (`VOIDCRAFT_BRIDGE_PORT`); voidcraft.world probes it.
- Env names: `VOIDCRAFT_BRIDGE_PLUGINS`, `VOIDCRAFT_BRIDGE_ALLOWED_HOSTS` (the legacy
  `TERMINAL_BRIDGE_ALLOWED_HOSTS` is unioned in), `VOIDCRAFT_LOCAL_LLM_URL`,
  `VOIDCRAFT_OLLAMA_URL`, `VOIDCRAFT_LOCAL_LLM_MODEL`.

## Commands

```bash
uv sync
uv run pytest                                # no model server needed; every test fakes one
uv run ruff check .                          # lint; the rule set is in pyproject.toml
uvx typos@1.36.3                             # spelling, code and docs
uv run voidcraft-world-bridge --port 7791    # a spare port, if a bridge already holds 7682
```
