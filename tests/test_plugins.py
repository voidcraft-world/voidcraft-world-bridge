"""Plugin host tests (voidcraft_world_bridge/plugins.py) — all socket-free.

The load-bearing guarantees: a stock bridge (no env var, no config) loads
nothing; a broken plugin dir NEVER stops the bridge from starting; a plugin
handler exception NEVER kills a request thread (500 instead); and the host is
the one applying route parsing — plugins only ever see their own subpath.
"""
import json
import sys
from pathlib import Path

from voidcraft_world_bridge.plugins import (
    DEFAULT_CONFIG_PATH,
    MAX_PLUGIN_BODY_BYTES,
    PluginContext,
    discover_plugin_dirs,
    dispatch_plugin,
    load_plugins,
    parse_plugin_path,
    plugin_body_limit,
    plugin_routes,
)


def _context(log=None):
    return PluginContext(port=7681, status_port=7682, default_session="voidcraft",
                         start_dir="/tmp", log=log or (lambda msg: None))


def _write_plugin(directory: Path, name: str = "demo", body: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "bridge_plugin.py").write_text(body or f'''
PLUGIN_NAME = "{name}"

class _Plugin:
    name = "{name}"
    def routes(self):
        return ["/plugin/{name}/ping"]
    def handle_get(self, subpath, query):
        return 200, {{"pong": subpath, "q": query.get("x", [None])[0]}}
    def handle_post(self, subpath, query, body):
        if subpath == "/boom":
            raise RuntimeError("kaboom")
        if subpath == "/raw":
            return 200, b"<html>hi</html>", "text/html"
        if subpath == "/bad":
            return "not a tuple"
        return 200, {{"ok": True, "body": body}}

def create_plugin(context):
    return _Plugin()
''')
    return directory


# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------
def test_parse_plugin_path():
    assert parse_plugin_path("/plugin/my-plugin/state") == ("my-plugin", "/state")
    assert parse_plugin_path("/plugin/my-plugin") == ("my-plugin", "/")
    assert parse_plugin_path("/plugin/my-plugin/ui/app.js") == ("my-plugin", "/ui/app.js")
    assert parse_plugin_path("/status") is None
    assert parse_plugin_path("/snapshot") is None
    assert parse_plugin_path("/plugin/") is None
    assert parse_plugin_path("/plugin/UPPER/x") is None  # names are lowercase tokens


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def test_env_var_wins_over_config(tmp_path):
    config = tmp_path / "plugins.json"
    config.write_text(json.dumps({"version": 1, "plugins": ["/from/config"]}))
    dirs = discover_plugin_dirs({"VOIDCRAFT_BRIDGE_PLUGINS": "/a:/b"}, config_path=config)
    assert dirs == [Path("/a"), Path("/b")]


def test_a_host_can_discover_under_its_own_names(tmp_path):
    # An embedding host keeps its own env var: the core's name must then be ignored,
    # or a stranger's plugin setting would leak into a host that never asked for it.
    config = tmp_path / "plugins.json"
    config.write_text(json.dumps({"version": 1, "plugins": ["/from/config"]}))
    env = {"HOST_PLUGINS": "/host", "VOIDCRAFT_BRIDGE_PLUGINS": "/core"}
    assert discover_plugin_dirs(env, config, env_var="HOST_PLUGINS") == [Path("/host")]
    assert discover_plugin_dirs({"VOIDCRAFT_BRIDGE_PLUGINS": "/core"}, config,
                                env_var="HOST_PLUGINS") == [Path("/from/config")]


def test_config_fallback_and_tolerance(tmp_path):
    config = tmp_path / "plugins.json"
    config.write_text(json.dumps({"version": 1, "plugins": ["/from/config", 42, ""]}))
    assert discover_plugin_dirs({}, config_path=config) == [Path("/from/config")]
    config.write_text("{corrupt")
    assert discover_plugin_dirs({}, config_path=config) == []
    assert discover_plugin_dirs({}, config_path=tmp_path / "missing.json") == []


def test_no_env_no_config_loads_nothing():
    # The default config path only matters if the dev created it; a missing file
    # must mean "no plugins", never an error.
    if not DEFAULT_CONFIG_PATH.exists():
        assert discover_plugin_dirs({}) == []


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def test_load_good_plugin(tmp_path):
    registry = load_plugins([_write_plugin(tmp_path / "demo")], _context())
    assert list(registry) == ["demo"]
    assert plugin_routes(registry) == {"demo": {"routes": ["/plugin/demo/ping"]}}


def test_broken_dirs_are_skipped_with_warning(tmp_path):
    warnings = []
    empty = tmp_path / "empty"
    empty.mkdir()
    syntax_error = tmp_path / "broken"
    syntax_error.mkdir()
    (syntax_error / "bridge_plugin.py").write_text("def nope(:\n")
    bad_name = _write_plugin(tmp_path / "badname", name="Bad Name!")
    raises = tmp_path / "raises"
    raises.mkdir()
    (raises / "bridge_plugin.py").write_text(
        'PLUGIN_NAME = "raises"\n\ndef create_plugin(context):\n    raise RuntimeError("nope")\n')
    good = _write_plugin(tmp_path / "good", name="good")

    registry = load_plugins([empty, syntax_error, bad_name, raises, good],
                            _context(log=warnings.append))
    assert list(registry) == ["good"]          # the bridge still starts
    assert len(warnings) == 4                  # one warning per broken dir


def test_a_sibling_package_imports_and_never_shadows_the_stdlib(tmp_path):
    # The entry module may import a package beside it (`from my_pkg… import`),
    # which is why the dir goes on sys.path at all — at the END. A plugin dir
    # ahead of the stdlib could hand a stray `json.py` to the whole process.
    directory = tmp_path / "pkg"
    (directory / "demo_pkg").mkdir(parents=True)
    (directory / "demo_pkg" / "__init__.py").write_text("")
    (directory / "demo_pkg" / "impl.py").write_text("def make():\n    return 'made'\n")
    (directory / "bridge_plugin.py").write_text(
        "from demo_pkg.impl import make\n"
        'PLUGIN_NAME = "pkg"\n\n'
        "class _Plugin:\n"
        '    name = "pkg"\n'
        "    def routes(self):\n        return []\n"
        "    def handle_get(self, subpath, query):\n        return 200, {'made': make()}\n"
        "    def handle_post(self, subpath, query, body):\n        return 405, {}\n\n"
        "def create_plugin(context):\n    return _Plugin()\n")
    registry = load_plugins([directory], _context())
    assert dispatch_plugin(registry, "GET", "pkg", "/", {}, None) == (200, {"made": "made"})
    assert sys.path[0] != str(directory)
    assert sys.path[-1] == str(directory)


def test_duplicate_name_first_wins(tmp_path):
    warnings = []
    first = _write_plugin(tmp_path / "first", name="demo")
    second = _write_plugin(tmp_path / "second", name="demo")
    registry = load_plugins([first, second], _context(log=warnings.append))
    assert list(registry) == ["demo"]
    assert any("already loaded" in w for w in warnings)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def test_dispatch_unknown_plugin_404():
    code, _ = dispatch_plugin({}, "GET", "nope", "/x", {}, None)
    assert code == 404
    code, _ = dispatch_plugin(None, "GET", "nope", "/x", {}, None)
    assert code == 404


def test_dispatch_get_and_post(tmp_path):
    registry = load_plugins([_write_plugin(tmp_path / "demo")], _context())
    code, payload = dispatch_plugin(registry, "GET", "demo", "/state", {"x": ["1"]}, None)
    assert (code, payload) == (200, {"pong": "/state", "q": "1"})
    code, payload = dispatch_plugin(registry, "POST", "demo", "/run", {}, {"env": "dev"})
    assert (code, payload) == (200, {"ok": True, "body": {"env": "dev"}})


def test_dispatch_raw_response_passthrough(tmp_path):
    registry = load_plugins([_write_plugin(tmp_path / "demo")], _context())
    code, body, content_type = dispatch_plugin(registry, "POST", "demo", "/raw", {}, None)
    assert (code, content_type) == (200, "text/html")
    assert body == b"<html>hi</html>"


def test_dispatch_handler_exception_becomes_500(tmp_path):
    registry = load_plugins([_write_plugin(tmp_path / "demo")], _context())
    code, payload = dispatch_plugin(registry, "POST", "demo", "/boom", {}, None)
    assert code == 500
    assert "RuntimeError" in payload["error"]


def test_dispatch_malformed_return_becomes_500(tmp_path):
    registry = load_plugins([_write_plugin(tmp_path / "demo")], _context())
    code, payload = dispatch_plugin(registry, "POST", "demo", "/bad", {}, None)
    assert code == 500
    assert "malformed" in payload["error"]


# ---------------------------------------------------------------------------
# Per-plugin body limit + header-aware dispatch
#
# Both exist for one kind of plugin: a proxy to an authenticated API. Its POST
# body is a proposal whose free-text field has no length limit, and its
# authentication is a bearer token the core would otherwise never hand to a
# handler. Both are opt-in and clamped, so no other plugin changes.
# ---------------------------------------------------------------------------
HEADER_PLUGIN = '''
PLUGIN_NAME = "hdr"

class _Plugin:
    name = "hdr"
    max_body_bytes = 5 * 1024 * 1024
    def routes(self):
        return ["/plugin/hdr/echo"]
    def handle_request(self, method, subpath, query, body, headers):
        return 200, {"method": method, "sub": subpath, "auth": headers.get("authorization")}

def create_plugin(context):
    return _Plugin()
'''


def test_body_limit_defaults_to_the_core_cap(tmp_path):
    registry = load_plugins([_write_plugin(tmp_path / "demo")], _context())
    assert plugin_body_limit(registry, "demo", 4096) == 4096
    # An unknown plugin gets the default too — the 404 comes later, not here.
    assert plugin_body_limit(registry, "nope", 4096) == 4096
    assert plugin_body_limit(None, "demo", 4096) == 4096


def test_body_limit_honours_an_opt_in_and_clamps_it(tmp_path):
    directory = tmp_path / "hdr"
    directory.mkdir()
    (directory / "bridge_plugin.py").write_text(HEADER_PLUGIN)
    registry = load_plugins([directory], _context())
    assert plugin_body_limit(registry, "hdr", 4096) == 5 * 1024 * 1024
    # Never below the core cap, and never past the ceiling.
    registry["hdr"].max_body_bytes = 10
    assert plugin_body_limit(registry, "hdr", 4096) == 4096
    registry["hdr"].max_body_bytes = 10**12
    assert plugin_body_limit(registry, "hdr", 4096) == MAX_PLUGIN_BODY_BYTES
    # A non-int (True is an int subclass, and a common accident) is ignored.
    registry["hdr"].max_body_bytes = True
    assert plugin_body_limit(registry, "hdr", 4096) == 4096


def test_handle_request_receives_headers(tmp_path):
    directory = tmp_path / "hdr"
    directory.mkdir()
    (directory / "bridge_plugin.py").write_text(HEADER_PLUGIN)
    registry = load_plugins([directory], _context())
    code, payload = dispatch_plugin(registry, "POST", "hdr", "/echo", {}, {},
                                    {"Authorization": "Bearer abc"})
    assert (code, payload) == (200, {"method": "POST", "sub": "/echo", "auth": "Bearer abc"})


def test_a_plugin_without_handle_request_never_sees_a_header(tmp_path):
    # Headers carry credentials. The old contract stays the default precisely so
    # that a plugin has to ASK before one is handed to it.
    registry = load_plugins([_write_plugin(tmp_path / "demo")], _context())
    code, payload = dispatch_plugin(registry, "GET", "demo", "/state", {}, None,
                                    {"Authorization": "Bearer abc"})
    assert code == 200
    assert "abc" not in json.dumps(payload)


def test_the_context_carries_no_profiles_unless_given():
    # A host that predates launch profiles hands plugins no registry, and a
    # plugin must find `None` there rather than an AttributeError, and may
    # branch on it.
    assert _context().profiles is None
    marker = object()
    context = PluginContext(port=7681, status_port=7682, default_session="voidcraft",
                            start_dir="/tmp", log=lambda msg: None, profiles=marker)
    assert context.profiles is marker
