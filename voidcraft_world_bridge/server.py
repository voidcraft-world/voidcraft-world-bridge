"""The bridge's HTTP server: guards, CORS, plugin dispatch and `/snapshot`.

`BridgeHandler` is the whole public surface a browser can reach. It answers
`GET /snapshot` (who this bridge is and which plugins it mounts) and
`/plugin/<name>/…`; every other path is 404. A host that needs more routes
SUBCLASSES it and names them in `GET_ROUTES` / `POST_ROUTES` — the guards then
run in front of those routes too, because they live in `do_GET`/`do_POST`, not in
each route. A route added later is covered by default rather than by
remembering.

Stdlib only, on purpose: this runs on a stranger's machine with loopback access
to their local model, and every dependency is one more thing they must trust.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

from voidcraft_world_bridge import __version__
from voidcraft_world_bridge.guards import host_allowed, is_no_cors_browser_request, origin_allowed
from voidcraft_world_bridge.plugins import (
    dispatch_plugin,
    parse_plugin_path,
    plugin_body_limit,
    plugin_routes,
)

NAME = "voidcraft-world-bridge"
DEFAULT_PORT = 7682
LOOPBACK = "127.0.0.1"

PROTOCOL = 1
"""The `/snapshot` shape. Additive-only: a client must ignore keys it does not
know, and a bump means a key changed meaning, never that one was added."""

MAX_BODY_BYTES = 4096
"""Default POST body cap. A plugin may raise its own (`plugin_body_limit`)."""


def bridge_block() -> dict:
    """The `bridge` key of `/snapshot`: what this server is. A host may add
    fields to it; it must not change these three."""
    return {"name": NAME, "version": __version__, "protocol": PROTOCOL}


class BridgeHandler(BaseHTTPRequestHandler):
    """Request handler for the bridge. Configure with `build_handler`, or
    subclass to add host routes (see the module docstring)."""

    server_version = NAME

    plugins: ClassVar[dict[str, object]] = {}
    # Host routes: path → name of a zero-argument method on the subclass. The
    # method reads `self.path` / `self.headers` itself and must send exactly one
    # response. Plugin paths and `/snapshot` are matched first.
    GET_ROUTES: ClassVar[Mapping[str, str]] = {}
    POST_ROUTES: ClassVar[Mapping[str, str]] = {}

    # --- what /snapshot says ------------------------------------------------

    def snapshot_payload(self) -> dict:
        """The `GET /snapshot` body. A host overriding this keeps `bridge_block()`
        under `bridge` and the `plugins` key, which clients use to detect what
        this bridge can do."""
        advertised = plugin_routes(self.plugins)
        return {
            "bridge": bridge_block(),
            # Omitted when none, so a client reads absence as "no plugins".
            **({"plugins": advertised} if advertised else {}),
        }

    # --- guards -------------------------------------------------------------

    def _host_guard(self) -> bool:
        """403 a request addressed to a Host we do not answer to.

        Runs BEFORE the Origin guard because it defeats an attack the Origin
        guard structurally cannot: under DNS rebinding the attacker's own
        hostname is the origin, so every same-origin check passes.
        """
        host = self.headers.get("Host")
        if host_allowed(host):
            return True
        self._send_json(403, {"error": f"host not allowed: {host}"})
        return False

    def _origin_allowed(self) -> bool:
        """403 an unrecognised browser origin. True when the request may proceed.

        False means a response has already been sent — same contract as
        `_read_json_object`. One method rather than a pasted check: a guard that
        gets pasted is a guard that eventually gets pasted incompletely.
        """
        origin = self.headers.get("Origin")
        if origin_allowed(origin):
            return True
        self._send_json(403, {"error": f"origin not allowed: {origin}"})
        return False

    # The name the request-level check in do_GET/do_POST goes by. Same function:
    # a route that re-checks (it acts on the machine) calls `_origin_allowed`.
    _origin_guard = _origin_allowed

    def _expensive_route_guard(self) -> bool:
        """Reject a no-cors browser GET on a route whose COST is the risk.

        Such a request carries no Origin, which `origin_allowed` reads as "a
        native client". A cross-origin `<img src="…">` is indistinguishable
        there, so a route that holds a thread or forks a process asks this too.
        """
        origin = self.headers.get("Origin")
        if not is_no_cors_browser_request(origin, self.headers.get("Sec-Fetch-Site")):
            return True
        self._send_json(403, {"error": "no-cors browser request not allowed on this route"})
        return False

    def _requesting_origin(self) -> str | None:
        """The caller's Origin header, for log lines. One accessor, so a log
        line cannot reach for a local that a guard extraction took away."""
        return self.headers.get("Origin")

    # --- responses ----------------------------------------------------------

    def _send_cors_headers(self) -> None:
        """Echo the caller's origin — never `*` — and only if it is allowed.

        An absent or rejected origin simply gets no CORS header: a native client
        does not read them, and a browser treats their absence as "you may not
        read this response", which is the intent.
        """
        origin = self.headers.get("Origin")
        if origin and origin_allowed(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            # The response body varies by origin, so it must not be cached
            # under one origin's key and replayed to another.
            self.send_header("Vary", "Origin")

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_plugin_response(self, response: tuple) -> None:
        """(status, dict) → JSON; (status, bytes, content_type) → raw (a plugin's
        own HTML page). Same origin-echoing CORS as _send_json."""
        if len(response) == 3:
            code, body, content_type = response
            self.send_response(code)
            self._send_cors_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(bytes(body))
            return
        self._send_json(response[0], response[1])

    def _content_length(self) -> int | None:
        """The declared body length: 0 when absent, None when it is not a number.

        `int()` on a garbage header used to raise inside the handler, which
        `http.server` answers by dropping the connection and printing a
        traceback. A malformed request is a 400, and the caller sends it.
        """
        raw = self.headers.get("Content-Length")
        if raw is None:
            return 0
        try:
            return int(raw)
        except ValueError:
            return None

    def _read_json_object(
        self, max_bytes: int = MAX_BODY_BYTES, *, require_object: bool = True
    ) -> dict | None:
        """Read and parse a JSON request body, or answer 400 and return None.

        THE CONTRACT: a None return means a response has ALREADY been sent, so
        every call site is

            payload = self._read_json_object()
            if payload is None:
                return

        and must not send anything of its own on that path.

        `require_object=False` is for a caller that accepts any JSON value and
        validates the shape itself.
        """
        length = self._content_length()
        if length is None or not 0 < length <= max_bytes:
            self._send_json(400, {"error": "bad content length"})
            return None
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "invalid JSON"})
            return None
        if require_object and not isinstance(payload, dict):
            self._send_json(400, {"error": "body must be a JSON object"})
            return None
        return payload

    # --- dispatch -----------------------------------------------------------

    def do_OPTIONS(self) -> None:
        if not self._host_guard():
            return
        origin = self.headers.get("Origin")
        if not origin_allowed(origin):
            # Fail the preflight rather than the actual request: the browser
            # then never sends the real one.
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        # Chrome's Private Network Access preflights a request from the deployed
        # https origin down to loopback; without this the fetch fails. Only ever
        # sent to an allowlisted origin — granting it to `*` is what would make
        # PNA useless as a second line of defence.
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def do_GET(self) -> None:
        # Guarded at the top, so a route added later is covered by default.
        # "Read-only" is not a reason to skip the check — see guards.py.
        if not self._host_guard():
            return
        if not self._origin_guard():
            return
        url = urlparse(self.path)
        plugin_ref = parse_plugin_path(url.path)
        if plugin_ref is not None:
            name, subpath = plugin_ref
            self._send_plugin_response(
                dispatch_plugin(self.plugins, "GET", name, subpath, parse_qs(url.query), None,
                                self.headers))
            return
        if url.path == "/snapshot":
            self._send_json(200, self.snapshot_payload())
            return
        route = self.GET_ROUTES.get(url.path)
        if route is not None:
            getattr(self, route)()
            return
        known = ", ".join(["/snapshot", *self.GET_ROUTES, "/plugin/<name>/…"])
        self._send_json(404, {"error": f"unknown path — GET {known}"})

    def do_POST(self) -> None:
        # Guarded at the top, same as do_GET. A write route is never exempt
        # because its payload "looks like telemetry": a cross-origin form POST is
        # a navigation, so it carries no preflight and CORS never gets a vote —
        # what matters is what the handler does with the body.
        if not self._host_guard():
            return
        if not self._origin_guard():
            return
        path = urlparse(self.path).path
        plugin_ref = parse_plugin_path(path)
        if plugin_ref is not None:
            self._dispatch_plugin_post(*plugin_ref)
            return
        route = self.POST_ROUTES.get(path)
        if route is not None:
            getattr(self, route)()
            return
        known = ", ".join([*self.POST_ROUTES, "/plugin/<name>/…"])
        self._send_json(404, {"error": f"unknown path — POST {known}"})

    def _dispatch_plugin_post(self, name: str, subpath: str) -> None:
        # Every plugin POST may act on the machine, so the Origin guard runs
        # again here — kept even though do_POST already checked, because a
        # plugin cannot be allowed to depend on a check it cannot see.
        if not self._origin_allowed():
            return
        # Per-plugin cap, defaulting to MAX_BODY_BYTES. Only a plugin that
        # declares `max_body_bytes` gets more, and never past
        # MAX_PLUGIN_BODY_BYTES — see plugin_body_limit.
        limit = plugin_body_limit(self.plugins, name, MAX_BODY_BYTES)
        length = self._content_length()
        body: dict | None = None
        if length is None or length > limit:
            self._send_json(400, {"error": "bad content length"})
            return
        if length > 0:
            try:
                parsed = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {"error": "invalid JSON"})
                return
            if not isinstance(parsed, dict):
                self._send_json(400, {"error": "body must be a JSON object"})
                return
            body = parsed
        self._send_plugin_response(
            dispatch_plugin(self.plugins, "POST", name, subpath,
                            parse_qs(urlparse(self.path).query), body, self.headers))

    def log_message(self, format: str, *args) -> None:
        pass  # clients poll every few seconds — per-request access logs are noise


def build_handler(plugins: dict[str, object] | None = None) -> type[BridgeHandler]:
    """A handler class bound to one plugin registry."""
    return type("Handler", (BridgeHandler,), {"plugins": dict(plugins or {})})


def make_server(port: int, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    """Bind loopback ONLY. There is no flag to bind anything wider: the guards
    assume the network cannot reach this port, and a LAN bind would hand every
    device on the network the local model. Raises OSError when the port is taken."""
    return ThreadingHTTPServer((LOOPBACK, port), handler)
