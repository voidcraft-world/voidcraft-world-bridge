from __future__ import annotations

import io
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from voidcraft_world_bridge.cli import main
from voidcraft_world_bridge.jobs import BackgroundJob, JobStore, QueueFull
from voidcraft_world_bridge.plugins import PluginContext, close_plugins, load_plugins, plugin_module_name, plugin_routes
from voidcraft_world_bridge.policy import CallerPolicy
from voidcraft_world_bridge.routing import Route, RoutedPlugin
from voidcraft_world_bridge.server import BridgeHandler
from voidcraft_world_bridge.testing import PluginClient


def test_policy_runs_before_handler_and_never_leaks_headers():
    calls = []
    plugin = RoutedPlugin("demo", [Route("POST", "/run", lambda q, b: (calls.append(b) or 200, {}))])
    plugin.request_policy = CallerPolicy(allowed_origins=frozenset({"https://voidcraft.world"}))
    client = PluginClient(plugin)
    for headers in ({"Origin": "https://elsewhere.example"}, {"Host": "remote.example"},
                    {"X-Forwarded-For": "127.0.0.1"}, {"Sec-Fetch-Site": "cross-site"}):
        assert client.request("POST", "/run", headers=headers)[0] == 403
    assert not calls
    assert client.request("POST", "/run", body={"x": 1}, headers={"Origin": "https://voidcraft.world"})[0] == 200
    assert calls == [{"x": 1}]


def test_routes_advertise_methods_and_dispatch_consistently():
    plugin = RoutedPlugin("demo", [Route("GET", "/state", lambda q, b: (200, {"ready": True}))])
    client = PluginClient(plugin)
    assert client.request("GET", "/state") == (200, {"ready": True})
    assert client.request("POST", "/state")[0] == 405
    assert client.request("GET", "/absent")[0] == 404
    advertised = plugin_routes({"demo": plugin})["demo"]
    assert advertised["routes"] == ["/plugin/demo/state"]
    assert advertised["operations"][0]["method"] == "GET"


@pytest.mark.parametrize("payload", [{"path": Path("example")}, {"number": float("nan")}, {"items": {1}}])
def test_non_json_return_becomes_500(payload):
    client = PluginClient(RoutedPlugin("bad", [Route("GET", "/", lambda q, b: (200, payload))]))
    assert client.request("GET", "/")[0] == 500


def test_negative_length_is_rejected_before_plugin_dispatch():
    replies = []
    handler = SimpleNamespace(plugins={}, headers={}, rfile=io.BytesIO(), _origin_allowed=lambda: True,
                              _content_length=lambda: -1, _send_json=lambda *args: replies.append(args))
    BridgeHandler._dispatch_plugin_post(handler, "demo", "/run")
    assert replies == [(400, {"error": "bad content length"})]


def test_invalid_factory_has_actionable_diagnostic(tmp_path):
    (tmp_path / "bridge_plugin.py").write_text('PLUGIN_NAME = "bad"\ndef create_plugin(context): return None\n')
    registry = load_plugins([tmp_path], PluginContext(status_port=7791, log=lambda msg: None))
    assert not registry
    assert registry.diagnostics[0]["error"] == "ValueError"
    assert main(["check-plugin", str(tmp_path)]) == 1


def test_validation_failure_closes_created_plugin(tmp_path):
    (tmp_path / "bridge_plugin.py").write_text('''
PLUGIN_NAME = "bad"
class Invalid:
    name = "wrong-name"
    def __init__(self, context): self.context = context
    def close(self): self.context.log("closed")
def create_plugin(context): return Invalid(context)
''')
    messages = []
    registry = load_plugins([tmp_path], PluginContext(status_port=7791, log=messages.append))
    assert not registry
    assert messages.count("closed") == 1
    assert registry.diagnostics[0]["error"] == "ValueError"
    assert plugin_module_name(tmp_path) not in sys.modules


def test_background_job_refuses_overlap_and_copies_result():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    result = {"ok": True, "rows": []}
    slot = BackgroundJob()

    def run():
        entered.set()
        assert release.wait(2)
        return result

    assert slot.start("sync", run)
    assert entered.wait(2)
    assert not slot.start("second", lambda: {})
    assert slot.busy_reasons() == ["sync"]
    release.set()
    # Joining this one worker avoids polling or sleeps in the assertion.
    for thread in threading.enumerate():
        if thread.name == "job-sync":
            thread.join(2)
    state = slot.snapshot()
    assert state == {"running": None, "last_run": result}
    state["last_run"]["rows"].append(1)
    assert slot.snapshot()["last_run"]["rows"] == []
    slot.close()
    assert not slot.start("closed", lambda: finished.set())


def test_queue_limits_are_configurable_and_close_rejects_new_work():
    jobs = JobStore(lambda request: {"text": "ok"}, start_worker=False, max_pending=1, max_kept=1)
    jobs.submit({})
    with pytest.raises(QueueFull):
        jobs.submit({})
    jobs.run_next()
    jobs.close()
    with pytest.raises(QueueFull):
        jobs.submit({})


def test_cleanup_continues_after_one_plugin_fails():
    calls = []

    def bad():
        raise RuntimeError("private detail")

    close_plugins({"bad": SimpleNamespace(close=bad), "good": SimpleNamespace(close=lambda: calls.append("good"))},
                  calls.append)
    assert calls[-1] == "good"
    assert "private detail" not in calls[0]
