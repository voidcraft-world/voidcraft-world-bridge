"""Compute hosts: other machines whose model server this bridge may hand a question to.

── WHY ANOTHER MACHINE AT ALL ────────────────────────────────────────────────
One model server answers one question at a time (`jobs.py`), so a second
machine on the same desk is a second lane: two Arena commanders decide at once
instead of in turn. The page still talks to this bridge only; which machine
decodes a question is a routing decision made here (`plugin.py`).

── WHY OVER SSH, AND ONLY AN ALIAS ───────────────────────────────────────────
The other machine's model server stays bound to ITS loopback. This bridge
reaches it through an SSH port forward (`tunnel.py`), so nothing listens on the
network, the traffic is encrypted, and the only credential is the key the user
already set up for `ssh <alias>`. A host is named by that alias and nothing
else: the bridge never stores an address, a user or a key, and an alias must be
an explicit `Host` entry in the user's own SSH config — so a page cannot point
the bridge at a machine the user has not already configured.

── THE FILE ──────────────────────────────────────────────────────────────────
`hosts.json` beside the bridge's `plugins.json`, or the path in
`VOIDCRAFT_LOCAL_LLM_HOSTS`:

    {"version": 1, "hosts": [{"name": "studio", "ssh": "studio", "kind": "ollama", "port": 11434}]}

A missing or corrupt file means "no other hosts", never an error — this
machine's model keeps working exactly as it did before the file existed.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from voidcraft_world_bridge.plugins import DEFAULT_CONFIG_PATH

LOCAL = "local"
"""The name this machine always goes by. It is never in the file and never removed."""

HOSTS_ENV = "VOIDCRAFT_LOCAL_LLM_HOSTS"
DEFAULT_HOSTS_PATH = DEFAULT_CONFIG_PATH.parent / "hosts.json"
SSH_CONFIG_PATH = Path.home() / ".ssh" / "config"

FILE_VERSION = 1
MAX_HOSTS = 8

KINDS = ("ollama", "openai")
DEFAULT_PORTS = {"ollama": 11434, "openai": 8080}

ALIAS_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
"""An SSH alias, as a `Host` line names it. Never starts with `-`, so it can
never be read as an option even before the `--` that precedes it in argv.
Always `fullmatch`: `$` would let a trailing newline through."""


@dataclass(frozen=True)
class RemoteHost:
    """One other machine: what to call it, the SSH alias that reaches it, and
    which model server listens on its loopback."""

    name: str
    ssh: str
    kind: str = "ollama"
    port: int = DEFAULT_PORTS["ollama"]


def hosts_path() -> Path:
    raw = os.environ.get(HOSTS_ENV, "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_HOSTS_PATH


def alias_problem(alias: object) -> str | None:
    """Why `alias` cannot name a host, or None. Shape only — `configured_aliases` is the other half."""
    if not isinstance(alias, str) or not ALIAS_RE.fullmatch(alias):
        return "an SSH alias is letters, digits, '.', '_' and '-', starting with a letter or digit"
    if alias.lower() == LOCAL:
        return f"'{LOCAL}' is this machine"
    return None


def parse_host(raw: object) -> RemoteHost | None:
    """One file entry, or None when any field is wrong. The file is user-edited."""
    if not isinstance(raw, dict):
        return None
    ssh = raw.get("ssh")
    name = raw.get("name", ssh)
    kind = raw.get("kind", "ollama")
    if alias_problem(ssh) or alias_problem(name) or kind not in KINDS:
        return None
    port = raw.get("port", DEFAULT_PORTS[kind])
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        return None
    return RemoteHost(name=str(name), ssh=str(ssh), kind=str(kind), port=port)


def load_hosts(path: Path | None = None) -> list[RemoteHost]:
    """Every valid entry, first of a name wins. Missing or corrupt → []."""
    try:
        data = json.loads((path or hosts_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = data.get("hosts") if isinstance(data, dict) else None
    hosts: list[RemoteHost] = []
    for raw in entries if isinstance(entries, list) else []:
        host = parse_host(raw)
        if host and all(h.name != host.name for h in hosts) and len(hosts) < MAX_HOSTS:
            hosts.append(host)
    return hosts


def save_hosts(hosts: Iterable[RemoteHost], path: Path | None = None) -> None:
    """Write atomically (a half-written file would read as "no hosts"), owner-only."""
    target = path or hosts_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_suffix(".json.tmp")
    payload = {"version": FILE_VERSION, "hosts": [asdict(h) for h in hosts]}
    fd = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(scratch, target)


def configured_aliases(path: Path | None = None) -> set[str]:
    """The literal aliases on `Host` lines of the user's SSH config.

    Wildcard patterns (`*`, `?`) and negations (`!`) are not aliases: `Host *`
    configures every host, which is exactly the "any machine at all" this check
    exists to refuse. `Include`d files are not followed; a host defined in one
    needs a plain `Host` line in the main config to be offered here.
    """
    try:
        text = (path or SSH_CONFIG_PATH).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    aliases: set[str] = set()
    for line in text.splitlines():
        words = line.split("#", 1)[0].replace("=", " ").split()
        if len(words) < 2 or words[0].lower() != "host":
            continue
        aliases.update(w for w in words[1:] if not any(c in w for c in "*?!"))
    return aliases
