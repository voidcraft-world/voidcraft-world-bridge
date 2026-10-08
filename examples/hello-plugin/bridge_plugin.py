"""The smallest complete bridge plugin: one page, one JSON route, GET and POST.

Run it on a spare port, from the repository root:

    VOIDCRAFT_BRIDGE_PLUGINS=$PWD/examples/hello-plugin voidcraft-world-bridge --port 7791

then, from another terminal (curl sends no Origin, which the guards treat as
a native client):

    curl http://127.0.0.1:7791/snapshot                           # lists "hello" and its routes
    curl 'http://127.0.0.1:7791/plugin/hello/greeting?name=you'
    curl -X POST -H 'Content-Type: application/json' -d '{"name": "you"}' \
         http://127.0.0.1:7791/plugin/hello/greeting
    open http://127.0.0.1:7791/plugin/hello/                     # the page, in a browser

Copy this directory to start your own plugin. The whole contract the host
expects is in `voidcraft_world_bridge/plugins.py`; this file exercises all of it
except `handle_request` (see the end). Plugins run inside the bridge's own
environment, so they are stdlib-only, like the bridge.
"""

from __future__ import annotations

PLUGIN_NAME = "hello"
"""A lowercase kebab token. It becomes the path: `/plugin/hello/...`."""

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>hello</title>
<p>A page served by a bridge plugin.</p>
<p><code>GET /plugin/hello/greeting?name=you</code> and
<code>POST /plugin/hello/greeting {"name": "you"}</code> answer in JSON.</p>
"""

ROUTES = ("/plugin/hello/", "/plugin/hello/greeting")
UNKNOWN = {"error": "unknown route — GET /, GET /greeting, POST /greeting"}


class HelloPlugin:
    name = PLUGIN_NAME

    max_body_bytes = 64 * 1024
    """Opt in to bodies past the host's 4 KB default (`server.MAX_BODY_BYTES`).
    The host enforces it and clamps it to its own ceiling; a plugin that POSTs
    anything real wants this line. Leave it out and the default applies."""

    def __init__(self, context) -> None:
        # Only `status_port` and `log` are guaranteed on every host; a plugin
        # that reads anything else treats an empty value as "not available".
        self._log = context.log
        self._port = context.status_port

    def routes(self) -> list[str]:
        """What `/snapshot` advertises, so a client can see what this bridge offers."""
        return list(ROUTES)

    def handle_get(self, subpath: str, query: dict[str, list[str]]) -> tuple:
        """`(status, dict)` is sent as JSON; `(status, bytes, content_type)` as-is."""
        if subpath == "/":
            return 200, PAGE.encode("utf-8"), "text/html; charset=utf-8"
        if subpath == "/greeting":
            name = (query.get("name") or ["world"])[0]
            return 200, {"greeting": f"hello, {name}"}
        return 404, dict(UNKNOWN)

    def handle_post(self, subpath: str, query: dict[str, list[str]],  # noqa: ARG002 — the handler signature is the contract
                    body: dict | None) -> tuple:
        """`body` is the parsed JSON object, or None when the request had none.
        The host has already applied its Origin guard and the body cap."""
        if subpath != "/greeting":
            return 404, dict(UNKNOWN)
        name = (body or {}).get("name")
        if not isinstance(name, str) or not name.strip():
            return 400, {"error": "bad_request", "message": "name must be a non-empty string"}
        return 200, {"greeting": f"hello, {name.strip()}"}


def create_plugin(context) -> HelloPlugin:
    """Called once at startup. Raising here skips the plugin; the bridge still starts."""
    context.log(f"hello: mounted at http://127.0.0.1:{context.status_port}/plugin/hello/")
    return HelloPlugin(context)


# Need the request headers — a bearer token, say? Define
#     handle_request(self, method, subpath, query, body, headers)
# INSTEAD of handle_get/handle_post, and the host calls that with the header
# names lowercased. It is not shown here because it replaces the pair above.
