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
| `shape.py` | `typed(value, kind, default)`: the one reader for a field of untrusted JSON, so mypy narrows and the `isinstance` idiom is written once. |
| `cli.py` | `voidcraft-world-bridge [status] [--port] [--version]`, argparse; the banner names the model server `local-llm` found. |
| `builtins.py` | The plugins every bridge mounts with no config (`local-llm`), and the one door for a host's agent runtimes. |
| `local_llm/plugin.py` | `/status`, `/chat` (202 + job), `/job`; body validation (it arrives from a browser). |
| `local_llm/runtimes.py` | Which model server answers: probe Ollama → LM Studio → llama.cpp, or the one configured URL. |
| `local_llm/ollama.py`, `openai_compat.py` | The two wire formats, same three functions; what does not translate to OpenAI's shape is in the latter's docstring. |
| `local_llm/models.py` | Which installed model answers; reads both Ollama tags and OpenAI-style ids. |
| `local_llm/jobs.py` | The bounded single-worker queue, and why generation is a job. |
| `local_llm/agents.py` | The agent-runtime contract. The package ships NONE; a host registers its own. |
| `scripts/release.py` | Cutting a release (below). Not shipped in the wheel; stdlib only all the same. |

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

## Releasing

A PyPI version is forever: it can be yanked, never re-uploaded. So a release is two steps, and
neither is done by hand.

```bash
uv run python scripts/release.py prepare patch   # or minor | major | X.Y.Z; then review + merge the PR
uv run python scripts/release.py tag             # on main, after the merge: the irreversible step
```

- `prepare` refuses an empty `## [Unreleased]`, so every user-visible change adds a line there
  in its own PR. It bumps `pyproject.toml`, moves `[Unreleased]` under the new version, rewrites
  the compare links, runs the whole CI ladder locally (plus the wheel smoke), and opens the PR.
- `tag` checks that `main` is clean and even with `origin`, that CI is green on HEAD, and that
  the version is not on PyPI yet. Then it asks, tags, pushes, watches `publish.yml`, and
  verifies: PyPI lists it, `uvx voidcraft-world-bridge@X.Y.Z` runs, and the GitHub Release has
  the wheel and the sdist. It is re-runnable: a tag already pushed skips to verification.
- Publishing is PyPI trusted publishing. Only `publish.yml`, in the `pypi` environment
  (which allows `v*` tags only), can upload. Renaming either breaks it until the publisher on
  PyPI is updated.
- The history is public. Commits and tags use a GitHub noreply address, set in this repo only;
  both steps refuse anything else.
- A host that embeds the bridge follows releases through `VOIDCRAFT_BRIDGE_CONSUMERS`
  (`os.pathsep`-separated project dirs) or `--consumer DIR`. `tag` ends by moving each
  consumer's `uv.lock` to the new version. The lock change is left uncommitted, and a version
  outside the consumer's declared range fails loudly instead of quietly keeping the old pin.
  `sync-consumers` does that step on its own. It also says when the bridge running on the
  default port is older than the release and needs a restart.
- Every step takes `--dry-run`.

## Commands

```bash
uv sync
uv run pytest                                # no model server needed; every test fakes one
uv run ruff check .                          # lint; the rule set is in pyproject.toml
uv run mypy                                  # types, the package only; config in pyproject.toml
uvx typos@1.36.3                             # spelling, code and docs
uv run voidcraft-world-bridge --port 7791    # a spare port, if a bridge already holds 7682
```
