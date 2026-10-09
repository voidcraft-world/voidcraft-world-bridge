"""The HTTP contract a browser sees, over a real loopback socket.

Each test serves a handler on an ephemeral port and speaks HTTP to it, because
the properties that matter — which requests get a 403, which headers come back,
what a host subclass inherits — only exist at that layer.
"""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from typing import ClassVar

from voidcraft_world_bridge import __version__
from voidcraft_world_bridge.server import NAME, PROTOCOL, BridgeHandler, build_handler, make_server


class _Echo:
    name = "echo"

    def routes(self):
        return ["/plugin/echo/hi"]

    def handle_get(self, subpath, query):
        return 200, {"subpath": subpath, "x": query.get("x", [None])[0]}

    def handle_post(self, subpath, query, body):
        return 200, {"got": body}


@contextmanager
def serving(handler):
    server = make_server(0, handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def request(port, method, path, *, headers=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    payload = json.dumps(body).encode() if body is not None else None
    all_headers = {"Content-Type": "application/json", **(headers or {})}
    conn.request(method, path, body=payload, headers=all_headers)
    response = conn.getresponse()
    raw = response.read()
    conn.close()
    try:
        parsed = json.loads(raw) if raw else None
    except ValueError:
        parsed = raw
    return response.status, dict(response.getheaders()), parsed


# --- /snapshot --------------------------------------------------------------


def test_snapshot_names_the_bridge_and_its_plugins():
    with serving(build_handler({"echo": _Echo()})) as port:
        status, _, body = request(port, "GET", "/snapshot")
    assert status == 200
    assert body["bridge"] == {"name": NAME, "version": __version__, "protocol": PROTOCOL}
    assert body["plugins"] == {"echo": {"routes": ["/plugin/echo/hi"]}}


def test_the_server_header_names_the_bridge_and_not_the_python_version():
    with serving(build_handler()) as port:
        _, headers, _ = request(port, "GET", "/snapshot")
    assert headers["Server"] == NAME


def test_snapshot_omits_plugins_when_none_are_mounted():
    # Clients read ABSENCE as "no plugins"; an empty map would be a second spelling.
    with serving(build_handler()) as port:
        _, _, body = request(port, "GET", "/snapshot")
    assert "plugins" not in body


def test_unknown_path_is_404_and_names_the_routes():
    with serving(build_handler()) as port:
        status, _, body = request(port, "GET", "/status")
    assert status == 404
    assert "/snapshot" in body["error"]


# --- plugin dispatch -----------------------------------------------------------


def test_plugin_get_and_post_dispatch():
    with serving(build_handler({"echo": _Echo()})) as port:
        status, _, body = request(port, "GET", "/plugin/echo/hi?x=1")
        assert (status, body) == (200, {"subpath": "/hi", "x": "1"})
        status, _, body = request(port, "POST", "/plugin/echo/hi", body={"a": 1})
        assert (status, body) == (200, {"got": {"a": 1}})


def test_plugin_post_body_over_the_default_cap_is_400():
    with serving(build_handler({"echo": _Echo()})) as port:
        status, _, _ = request(port, "POST", "/plugin/echo/hi", body={"pad": "x" * 5000})
    assert status == 400


def test_a_content_length_that_is_not_a_number_is_400_on_every_body_route():
    # Was a ValueError inside the handler: http.server printed a traceback and
    # dropped the connection. Garbage in a header is the client's problem.
    bad = {"Content-Length": "abc"}
    with serving(build_handler({"echo": _Echo()})) as port:
        status, _, body = request(port, "POST", "/plugin/echo/hi", headers=bad)
        assert (status, body["error"]) == (400, "bad content length")
    with serving(_HostHandler) as port:
        status, _, body = request(port, "POST", "/act", headers=bad)
        assert (status, body["error"]) == (400, "bad content length")


def test_unknown_plugin_is_404():
    with serving(build_handler()) as port:
        status, _, _ = request(port, "GET", "/plugin/nope/x")
    assert status == 404


# --- guards -----------------------------------------------------------------


def test_voidcraft_origin_is_served_and_echoed_never_wildcarded():
    with serving(build_handler()) as port:
        status, headers, _ = request(port, "GET", "/snapshot",
                                     headers={"Origin": "https://voidcraft.world"})
    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == "https://voidcraft.world"
    assert headers["Vary"] == "Origin"


def test_foreign_origin_is_refused_on_every_method():
    evil = {"Origin": "https://evil.example"}
    with serving(build_handler({"echo": _Echo()})) as port:
        assert request(port, "GET", "/snapshot", headers=evil)[0] == 403
        assert request(port, "GET", "/plugin/echo/hi", headers=evil)[0] == 403
        assert request(port, "POST", "/plugin/echo/hi", headers=evil, body={})[0] == 403
        assert request(port, "OPTIONS", "/plugin/echo/hi", headers=evil)[0] == 403


def test_rebinding_host_is_refused_even_from_an_allowed_origin():
    headers = {"Host": "rebind.evil.com:7682", "Origin": "https://voidcraft.world"}
    with serving(build_handler()) as port:
        status, _, body = request(port, "GET", "/snapshot", headers=headers)
    assert status == 403
    assert "host not allowed" in body["error"]


def test_preflight_grants_private_network_access_only_to_allowed_origins():
    with serving(build_handler()) as port:
        status, headers, _ = request(port, "OPTIONS", "/plugin/x/y",
                                     headers={"Origin": "https://voidcraft.world"})
    assert status == 204
    assert headers["Access-Control-Allow-Private-Network"] == "true"
    assert headers["Access-Control-Allow-Origin"] == "https://voidcraft.world"


# --- host subclasses ----------------------------------------------------------


class _HostHandler(BridgeHandler):
    # The spelling a host uses: ClassVar, so the routes are the class's, not an instance's.
    GET_ROUTES: ClassVar[Mapping[str, str]] = {"/extra": "_handle_extra"}
    POST_ROUTES: ClassVar[Mapping[str, str]] = {"/act": "_handle_act"}

    def _handle_extra(self):
        self._send_json(200, {"extra": True})

    def _handle_act(self):
        payload = self._read_json_object()
        if payload is None:
            return
        self._send_json(200, {"acted": payload})


def test_host_routes_are_served():
    with serving(_HostHandler) as port:
        assert request(port, "GET", "/extra")[2] == {"extra": True}
        assert request(port, "POST", "/act", body={"k": 1})[2] == {"acted": {"k": 1}}


def test_host_routes_inherit_the_guards_without_asking():
    # The point of putting the guards in do_GET/do_POST: a host route cannot forget them.
    evil = {"Origin": "https://evil.example"}
    with serving(_HostHandler) as port:
        assert request(port, "GET", "/extra", headers=evil)[0] == 403
        assert request(port, "POST", "/act", headers=evil, body={})[0] == 403
        assert request(port, "GET", "/extra", headers={"Host": "evil.example"})[0] == 403


def test_make_server_binds_loopback_only():
    server = make_server(0, build_handler())
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()
