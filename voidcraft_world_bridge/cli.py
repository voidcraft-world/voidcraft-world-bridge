"""`voidcraft-world-bridge` — run the bridge in the foreground, or ask one how it is.

    voidcraft-world-bridge              serve on 127.0.0.1:7682 until Ctrl-C
    voidcraft-world-bridge status       is one running here, and what does it mount?
    voidcraft-world-bridge --version
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from voidcraft_world_bridge import __version__
from voidcraft_world_bridge.builtins import LOCAL_LLM, builtin_plugins
from voidcraft_world_bridge.plugins import Plugin, PluginContext, discover_plugin_dirs, load_plugins
from voidcraft_world_bridge.server import DEFAULT_PORT, LOOPBACK, NAME, build_handler, make_server
from voidcraft_world_bridge.shape import typed

PORT_ENV = "VOIDCRAFT_BRIDGE_PORT"


def _log(message: str) -> None:
    print(message, flush=True)


def _default_port() -> int:
    raw = os.environ.get(PORT_ENV)
    try:
        return int(raw) if raw else DEFAULT_PORT
    except ValueError:
        return DEFAULT_PORT


def serve(port: int) -> int:
    """Load plugins, bind loopback, serve until Ctrl-C. Exit code: 0 on Ctrl-C,
    1 when the port is taken."""
    context = PluginContext(status_port=port, log=_log)
    plugins = load_plugins(discover_plugin_dirs(os.environ), context, preloaded=builtin_plugins(context))
    try:
        server = make_server(port, build_handler(plugins))
    except OSError as err:
        _log(f"{NAME}: cannot listen on {LOOPBACK}:{port} ({err.strerror or err}).")
        _log(f"  Is another bridge already running? Check with: {NAME} status")
        return 1
    _log(f"{NAME} {__version__} — listening on http://{LOOPBACK}:{port}")
    for name in plugins:
        _log(f"  plugin: {name}  → http://{LOOPBACK}:{port}/plugin/{name}/…")
    if LOCAL_LLM in plugins:
        _log(f"  local model:  {describe_local_model(plugins[LOCAL_LLM])}")
    _log("  Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log(f"\n{NAME}: stopped")
    finally:
        server.server_close()
    return 0


def describe_local_model(plugin: Plugin) -> str:
    """One line on what `local-llm` found, read through its own `/status` route,
    so the banner and voidcraft.world can never disagree."""
    _, status = plugin.handle_get("/status", {})
    if not status.get("reachable"):
        return ("none found — start Ollama or LM Studio (or llama-server), "
                "or set VOIDCRAFT_LOCAL_LLM_URL; the bridge looks again on every request")
    picked = status.get("picked_model") or "no chat model installed yet"
    return f"{status.get('runtime_label')} at {status.get('runtime_url')} → {picked}"


def fetch_snapshot(port: int, timeout: float = 2.0) -> dict | None:
    """GET /snapshot from a bridge on this machine, or None if nothing answers."""
    try:
        with urllib.request.urlopen(f"http://{LOOPBACK}:{port}/snapshot", timeout=timeout) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def status(port: int) -> int:
    """Print what a running bridge reports. Exit 1 when none answers."""
    snapshot = fetch_snapshot(port)
    if snapshot is None:
        _log(f"{NAME}: nothing answering on {LOOPBACK}:{port}")
        return 1
    bridge = typed(snapshot.get("bridge"), dict, {})
    plugins = typed(snapshot.get("plugins"), dict, {})
    _log(f"{bridge.get('name', 'a bridge')} {bridge.get('version', '?')} — ONLINE on {LOOPBACK}:{port}")
    _log(f"  plugins: {', '.join(sorted(plugins)) or '(none)'}")
    return 0


def _add_port(parser: argparse.ArgumentParser, *, default: object) -> None:
    parser.add_argument("--port", type=int, default=default,
                        help=f"loopback port (default {DEFAULT_PORT}, or ${PORT_ENV})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=NAME, description="Let voidcraft.world reach your machine.")
    parser.add_argument("--version", action="version", version=f"{NAME} {__version__}")
    _add_port(parser, default=_default_port())
    commands = parser.add_subparsers(dest="command")
    # `status --port N` reads as naturally as `--port N status`, so the
    # subcommand takes the flag too. Its default is SUPPRESS: a subparser's
    # default would otherwise overwrite a value the root parser already read.
    _add_port(commands.add_parser("status", help="report on a bridge running here"),
              default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.command == "status":
        return status(args.port)
    return serve(args.port)


if __name__ == "__main__":
    sys.exit(main())
