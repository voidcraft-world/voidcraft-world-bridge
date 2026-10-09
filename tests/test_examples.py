"""The example plugin is loaded, dispatched and served exactly as a user's would
be, so it cannot drift from the contract it documents."""

from __future__ import annotations

from pathlib import Path

from test_server import request, serving

from voidcraft_world_bridge.plugins import PluginContext, dispatch_plugin, load_plugins, plugin_routes
from voidcraft_world_bridge.server import build_handler

HELLO = Path(__file__).resolve().parents[1] / "examples" / "hello-plugin"


def _load(log=None):
    context = PluginContext(status_port=7791, log=log or (lambda msg: None))
    return load_plugins([HELLO], context)


def test_hello_loads_and_advertises_its_routes():
    lines = []
    registry = _load(lines.append)
    assert list(registry) == ["hello"]
    assert plugin_routes(registry) == {"hello": {"routes": ["/plugin/hello/", "/plugin/hello/greeting"]}}
    assert lines == ["hello: mounted at http://127.0.0.1:7791/plugin/hello/"]


def test_hello_dispatch_covers_the_contract():
    registry = _load()
    assert dispatch_plugin(registry, "GET", "hello", "/greeting", {"name": ["you"]}, None) == (
        200, {"greeting": "hello, you"})
    assert dispatch_plugin(registry, "GET", "hello", "/greeting", {}, None) == (200, {"greeting": "hello, world"})
    assert dispatch_plugin(registry, "POST", "hello", "/greeting", {}, {"name": "you"}) == (
        200, {"greeting": "hello, you"})
    assert dispatch_plugin(registry, "POST", "hello", "/greeting", {}, {"name": 7})[0] == 400
    assert dispatch_plugin(registry, "GET", "hello", "/nope", {}, None)[0] == 404
    assert dispatch_plugin(registry, "POST", "hello", "/nope", {}, None)[0] == 404
    code, body, content_type = dispatch_plugin(registry, "GET", "hello", "/", {}, None)
    assert (code, content_type) == (200, "text/html; charset=utf-8")
    assert b"<title>hello</title>" in body


def test_hello_over_a_real_socket():
    origin = {"Origin": "https://voidcraft.world"}
    with serving(build_handler(_load())) as port:
        status, headers, body = request(port, "GET", "/plugin/hello/greeting?name=you", headers=origin)
        assert (status, body) == (200, {"greeting": "hello, you"})
        assert headers["Access-Control-Allow-Origin"] == "https://voidcraft.world"
        # 10 KB is past the core's 4 KB cap: the plugin's `max_body_bytes` opt-in is what lets it in.
        status, _, body = request(port, "POST", "/plugin/hello/greeting", headers=origin,
                                  body={"name": "you", "pad": "x" * 10_000})
        assert (status, body) == (200, {"greeting": "hello, you"})
        status, headers, body = request(port, "GET", "/plugin/hello/")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert b"<title>hello</title>" in body
