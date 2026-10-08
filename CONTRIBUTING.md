# Contributing

Issues and pull requests are welcome. Small, focused changes land fastest.

## The two rules (both are tests)

1. **Stdlib only.** The package imports nothing outside the standard library. A dependency is a
   `pyproject.toml` decision, never an import that slips in — `tests/test_purity.py` walks every
   import. This code runs on a stranger's machine with loopback access to their model; every
   dependency is one more thing they must trust.
2. **Nothing private.** The package was carved out of a private codebase. No local paths, email
   addresses or pointers back into that codebase, in code, docs or tests. The same test scans every
   shipped file for the shapes a leak takes.

## Working on it

```bash
uv sync
uv run pytest                                # no model server needed; every test fakes one
uv run ruff check .                          # lint
uvx typos@1.36.3                             # spelling
uv run voidcraft-world-bridge --port 7791    # a spare port, if a bridge already holds 7682
```

CI runs the same suite on Linux and macOS across every supported Python. `CLAUDE.md` is the guide
for coding agents and the map of the modules; `AGENTS.md` is a symlink to it.

## Style

- `snake_case` for functions and variables; a public exception keeps the name clients already
  catch (`RuntimeDown`, `JobFailure`), so do not rename one for a lint rule.
- A comment explains *why*, once, next to the mechanism. Everywhere else, a one-line pointer.
- The contracts clients depend on (`/snapshot`, `/plugin/local-llm/status`, the env names) are
  additive-only. Add keys; never change what an existing key means.
- A fix ships with the test that would have caught it.

## Conduct

Be kind and on topic. This repository keeps no contact address anywhere (see rule 2), so reports
about behaviour go through GitHub's own reporting tools, and the maintainers moderate issues and
pull requests directly.
