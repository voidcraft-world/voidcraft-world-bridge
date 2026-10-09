"""What this package may contain, checked rather than trusted.

Two promises a stranger installing it relies on:

1. **Stdlib only.** Every import is the standard library or this package. A new
   dependency is a decision, made in pyproject.toml, never one that slips in
   through an import.
2. **Nothing private.** This package was carved out of a private codebase. A path,
   an account name or a pointer back into that codebase would leak it — so the
   source is scanned for the shapes those take.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "voidcraft_world_bridge"
# The examples run inside the bridge's environment, and the release script runs from a
# bare checkout, so both keep the same promise.
SOURCES = sorted([*PACKAGE.rglob("*.py"), *(ROOT / "examples").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")])
# Everything the public repo carries — docs and tests included: a leak in a README or
# a test comment is still a leak. This file is the one exception, since it has to
# spell out the shapes it looks for.
SHIPPED = sorted(
    path for pattern in ("*.py", "*.md", "*.toml", "*.yml", "LICENSE", ".gitignore")
    for path in ROOT.rglob(pattern)
    if not {".venv", "__pycache__", ".pytest_cache"} & set(path.relative_to(ROOT).parts)
    and path.resolve() != Path(__file__).resolve()
)

# Shapes a leak takes. Each is a pattern, not a list of secrets: the secrets
# themselves must never be written down here either.
FORBIDDEN = {
    "a home directory path": re.compile(r"/Users/|/home/[a-z]"),
    "an email address": re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.IGNORECASE),
    "a private params dir": re.compile(r"VOIDCRAFT_PARAMS|voidcraft-params"),
    "a personal tool": re.compile(r"trading|plaid|garmin|steward|robinhood", re.IGNORECASE),
    "the private bridge's internals": re.compile(
        r"terminal_bridge|bridge-plugins|front-end/src|mcp-local|mcp_local|MCP proxy"),
    "a Claude account dir": re.compile(r"\.claude-[a-z]"),
    "a word about the private side": re.compile(r"\bfounder|macos app|menu-bar", re.IGNORECASE),
}


def test_the_package_has_sources():
    assert PACKAGE.is_dir() and SOURCES, PACKAGE


def test_agents_md_is_the_agent_guide_itself():
    """One guide for every coding agent: AGENTS.md points at CLAUDE.md, never a copy that drifts."""
    agents = ROOT / "AGENTS.md"
    assert agents.is_symlink() and agents.readlink() == Path("CLAUDE.md"), agents


def test_imports_are_stdlib_or_this_package():
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    offenders = []
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if root not in stdlib and root != "voidcraft_world_bridge":
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, offenders


def test_no_private_strings():
    assert any(path.suffix == ".md" for path in SHIPPED), SHIPPED  # the docs are scanned too
    hits = []
    for path in SHIPPED:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for what, pattern in FORBIDDEN.items():
                if pattern.search(line):
                    hits.append(f"{path.name}:{lineno}: {what}: {line.strip()[:80]}")
    assert not hits, "\n".join(hits)
