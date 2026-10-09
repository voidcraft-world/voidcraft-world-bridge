"""Plugin host: mount capabilities under /plugin/<name>/ on the bridge's port.

The bridge core stays generic; everything it can actually DO lives in a plugin,
discovered at startup and mounted as HTTP routes. Discovery is explicit — env
`VOIDCRAFT_BRIDGE_PLUGINS` (colon-separated absolute dirs) wins, else
`~/.config/voidcraft-world-bridge/plugins.json`
({"version": 1, "plugins": [dirs]}) — so a bridge with neither configured loads
only what it ships with. A host embedding this core may pass its own env var
and config path to `discover_plugin_dirs`.

Entry contract (per plugin dir):
- `<dir>/bridge_plugin.py` defines module-level `PLUGIN_NAME` (lowercase
  kebab token) and `create_plugin(context: PluginContext) -> plugin`.
- The plugin object is duck-typed: `name`, `routes() -> list[str]`,
  `handle_get(subpath, query)` and `handle_post(subpath, query, body)`, each
  returning `(status, dict)` for JSON or `(status, bytes, content_type)` for
  raw responses (e.g. a served HTML page).
- Plugins execute in the bridge's venv → stdlib-only imports. Heavy work must
  shell out to the plugin's own tooling; handlers must not block long (any
  subprocess call needs a timeout).
- The host applies the bridge's Origin guard to every plugin POST before
  dispatch (plugins cannot weaken it) and wraps handler exceptions in a 500 —
  a broken plugin must never take the bridge down, at load OR request time.

JSON config (not TOML): tomllib is stdlib only from 3.11 and the bridge
supports 3.10. A corrupt or missing file loads nothing rather than failing.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

PLUGIN_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
PLUGIN_PATH_RE = re.compile(r"^/plugin/([a-z0-9][a-z0-9-]{0,31})(/.*)?$")
ENTRY_MODULE = "bridge_plugin.py"
PLUGINS_ENV = "VOIDCRAFT_BRIDGE_PLUGINS"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "voidcraft-world-bridge" / "plugins.json"

# Hard ceiling on a plugin's POST body, whatever it asks for. The default stays
# the core's 4 KB (server.MAX_BODY_BYTES) — this is only the most a plugin may
# raise itself to. It exists because a plugin that proxies a richer API (a
# proposal whose free-text field has no length limit of its own) routinely
# carries bodies past 4 KB. Raising the core cap for everybody to serve that
# case would be the wrong trade; opting in per plugin, with a ceiling, is the
# right one.
MAX_PLUGIN_BODY_BYTES = 5 * 1024 * 1024


class Plugin(Protocol):
    """What the host calls on a mounted plugin. Duck-typed: nothing inherits this,
    and the host never checks it at runtime.

    Two opt-ins are NOT members, on purpose — the host reads them with `getattr`,
    so a plugin written before they existed never has to know: an int
    `max_body_bytes` (`plugin_body_limit`), and `handle_request(method, subpath,
    query, body, headers)`, which REPLACES the get/post pair for a plugin that
    needs the request headers (`dispatch_plugin`).
    """

    @property
    def name(self) -> str: ...
    def routes(self) -> list[str]: ...
    def handle_get(self, subpath: str, query: dict[str, list[str]]) -> tuple: ...
    def handle_post(self, subpath: str, query: dict[str, list[str]], body: dict | None) -> tuple: ...


@dataclass(frozen=True)
class PluginContext:
    """What the host tells a plugin about the bridge it's mounted on.

    Only `status_port` and `log` are guaranteed. The rest describe a host that
    also runs terminals, and read as empty on one that does not — a plugin that
    needs them must treat the empty value as "not available here".
    """

    status_port: int                # the port plugin routes are served on
    log: Callable[[str], None]      # startup/diagnostic echo
    port: int = 0                   # a terminal server's port, 0 when the host has none
    default_session: str = ""
    start_dir: str = ""             # the host's start directory, "" when it has none
    # An optional launch-profile registry supplied by the host, or None.
    # Duck-typed on purpose — plugins cannot import the host, so they read it
    # with `getattr(context, "profiles", None)` and call only its plain-value
    # methods.
    profiles: object | None = None


def discover_plugin_dirs(env: Mapping[str, str],
                         config_path: Path | None = None,
                         *, env_var: str = PLUGINS_ENV) -> list[Path]:
    """Plugin directories to load. Env var wins; config file is the fallback;
    neither configured (or a corrupt config) → no plugins, never an error."""
    raw = env.get(env_var)
    if raw:
        return [Path(p).expanduser() for p in raw.split(":") if p.strip()]
    path = config_path or DEFAULT_CONFIG_PATH
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    dirs = data.get("plugins") if isinstance(data, dict) else None
    if not isinstance(dirs, list):
        return []
    return [Path(d).expanduser() for d in dirs if isinstance(d, str) and d.strip()]


def load_plugins(dirs: list[Path], context: PluginContext,
                 preloaded: dict[str, Plugin] | None = None) -> dict[str, Plugin]:
    """Import each plugin dir's entry module and build the name → plugin registry.

    ANY failure (missing entry, bad name, import error, create_plugin raising)
    logs a warning and skips that dir — the bridge must never fail to start
    because of a plugin. Duplicate names: first wins, and `preloaded` (the
    built-ins) come first.
    """
    registry: dict[str, Plugin] = dict(preloaded or {})
    for directory in dirs:
        try:
            entry = directory / ENTRY_MODULE
            if not entry.is_file():
                raise FileNotFoundError(f"no {ENTRY_MODULE} in {directory}")
            # The plugin's own package (a sibling of its entry module) must be
            # importable from the entry module's point of view. APPENDED, never
            # inserted first: a plugin dir ahead of the stdlib would let a stray
            # `json.py` in it hijack that name for the whole process and every
            # later plugin. Appended, a colliding name fails THIS plugin's import
            # loudly and the bridge logs it and moves on. Sibling packages must
            # still be uniquely named — two plugins sharing one share sys.modules.
            if str(directory) not in sys.path:
                sys.path.append(str(directory))
            module_name = "bridge_plugin_" + re.sub(r"[^a-z0-9]+", "_", directory.name.lower())
            spec = importlib.util.spec_from_file_location(module_name, entry)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {entry}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            name = getattr(module, "PLUGIN_NAME", None)
            if not isinstance(name, str) or not PLUGIN_NAME_RE.fullmatch(name):
                raise ValueError(f"invalid PLUGIN_NAME: {name!r}")
            if name in registry:
                context.log(f"bridge: plugin '{name}' already loaded — skipping {directory}")
                continue
            registry[name] = module.create_plugin(context)
        except Exception as exc:  # noqa: BLE001 — a broken plugin must never stop the bridge
            context.log(
                f"bridge: WARNING — plugin at {directory} failed to load:"
                f" {type(exc).__name__}: {exc}"
            )
    return registry


def parse_plugin_path(path: str) -> tuple[str, str] | None:
    """"/plugin/my-plugin/state" → ("my-plugin", "/state");
    None for any non-plugin path."""
    match = PLUGIN_PATH_RE.match(path)
    if match is None:
        return None
    return match.group(1), match.group(2) or "/"


def plugin_body_limit(registry: Mapping[str, Plugin] | None, name: str,
                      default: int) -> int:
    """How large a POST body this plugin accepts, clamped to MAX_PLUGIN_BODY_BYTES.

    A plugin opts in by exposing an int `max_body_bytes`; anything absent,
    non-integer or smaller than the default leaves the core limit in place. The
    host stays the one that enforces it — a plugin never reads the socket."""
    plugin = (registry or {}).get(name)
    requested = getattr(plugin, "max_body_bytes", None)
    if not isinstance(requested, int) or isinstance(requested, bool):
        return default
    return max(default, min(requested, MAX_PLUGIN_BODY_BYTES))


def dispatch_plugin(registry: Mapping[str, Plugin] | None, method: str, name: str,
                    subpath: str, query: dict[str, list[str]],
                    body: dict | None,
                    headers: Mapping[str, str] | None = None) -> tuple:
    """Route one request to a plugin handler. Socket-free and exception-proof:
    unknown plugin → 404, handler exception → 500, malformed handler return
    value → 500. Returns the same (status, dict) / (status, bytes, ctype)
    shapes the handlers produce.

    A plugin that needs the REQUEST HEADERS — one that proxies an authenticated
    API, where the caller's bearer token is what authenticates it — defines
    `handle_request(method, subpath, query, body, headers)` instead of the
    handle_get/handle_post pair, and the host calls that. Additive: a plugin
    without it never sees a header, which is the right default (headers carry
    credentials, and most plugins have no business reading them).

    Header NAMES arrive lowercased. `http.server` hands over an
    `email.message.Message`, which is case-insensitive; a plain `dict()` of it is
    not, and a plugin reading `headers["authorization"]` off a request that spelt
    it `Authorization` would silently find nothing. Normalising once here is
    cheaper than every plugin re-implementing the lookup."""
    plugin = (registry or {}).get(name)
    if plugin is None:
        return 404, {"error": f"unknown plugin: {name}"}
    try:
        handle_request = getattr(plugin, "handle_request", None)
        if callable(handle_request):
            lowered = {str(k).lower(): v for k, v in (headers or {}).items()}
            response = handle_request(method, subpath, query, body, lowered)
        elif method == "GET":
            response = plugin.handle_get(subpath, query)
        else:
            response = plugin.handle_post(subpath, query, body)
    except Exception as exc:  # noqa: BLE001 — a plugin bug must never kill the bridge
        return 500, {"error": f"plugin '{name}' error: {type(exc).__name__}"}
    if (isinstance(response, tuple) and len(response) == 2
            and isinstance(response[0], int) and isinstance(response[1], dict)):
        return response
    if (isinstance(response, tuple) and len(response) == 3
            and isinstance(response[0], int) and isinstance(response[1], (bytes, bytearray))
            and isinstance(response[2], str)):
        return response
    return 500, {"error": f"plugin '{name}' returned a malformed response"}


def plugin_routes(registry: Mapping[str, Plugin] | None) -> dict:
    """The additive /snapshot advertisement: {name: {"routes": [...]}}."""
    advertised: dict[str, dict] = {}
    for name, plugin in (registry or {}).items():
        try:
            routes = [str(r) for r in plugin.routes()]
        except Exception:  # noqa: BLE001 — advertisement must never break /snapshot
            routes = []
        advertised[name] = {"routes": routes}
    return advertised
