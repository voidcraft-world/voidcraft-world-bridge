from __future__ import annotations

import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from voidcraft_world_bridge.services import ManagedService


def service(tmp_path, **kwargs):
    return ManagedService(lambda port: None, environment=lambda port, key: {},
                          log_path=tmp_path / "service.log", **kwargs)


def test_unavailable_command_backs_off_and_close_prevents_restart(tmp_path):
    runtime = service(tmp_path, clock=lambda: 10)
    assert runtime.ensure_started() == "unavailable"
    runtime._command = lambda port: (_ for _ in ()).throw(AssertionError("must back off"))
    assert runtime.ensure_started() == "unavailable"
    runtime.close()
    assert runtime.ensure_started() == "stopped"


def test_stale_health_worker_cannot_stop_replacement(tmp_path):
    runtime = service(tmp_path, start_timeout_seconds=0)
    old = SimpleNamespace(poll=lambda: None)
    new = SimpleNamespace(poll=lambda: None)
    runtime._proc, runtime._state = new, "starting"
    runtime._await_health(old)
    runtime.stop(expected_proc=old)
    assert runtime._proc is new and runtime._state == "starting"


def test_idle_reaper_leaves_active_request_alone(tmp_path):
    runtime = service(tmp_path, clock=lambda: 1000, idle_stop_seconds=1)
    runtime._state, runtime._last_used, runtime._active = "ready", 0, 1
    assert runtime.reap_if_idle() is False


def test_bad_health_payload_is_not_ready(tmp_path, monkeypatch):
    runtime = service(tmp_path)
    monkeypatch.setattr(runtime, "request", lambda *args, **kwargs: (200, b"[]"))
    assert runtime._get_health() is None


def test_stop_terminates_only_owned_child(tmp_path):
    calls = []
    runtime = service(tmp_path)
    runtime._proc = SimpleNamespace(poll=lambda: None, terminate=lambda: calls.append("terminate"),
                                    wait=lambda timeout: calls.append(timeout))
    runtime.close()
    runtime.close()
    assert calls == ["terminate", 10]


def test_stopped_or_exited_service_does_not_send_to_stale_port(tmp_path, monkeypatch):
    calls = []
    runtime = service(tmp_path)
    monkeypatch.setattr(runtime._http, "open", lambda *a, **kw: calls.append(a))
    runtime._port, runtime._key, runtime._state = 7799, "old-key", "ready"
    runtime._proc = SimpleNamespace(poll=lambda: 1, returncode=1)
    assert runtime.request("POST", "/work", {"content": "example"}, 2) is None
    runtime.close()
    assert runtime.request("GET", "/health", None, 2) is None
    assert not calls


def test_stale_probe_cannot_contact_replacement(tmp_path, monkeypatch):
    calls = []
    runtime = service(tmp_path)
    monkeypatch.setattr(runtime._http, "open", lambda *a, **kw: calls.append(a))
    old = SimpleNamespace(poll=lambda: None)
    runtime._proc = SimpleNamespace(poll=lambda: None)
    runtime._port, runtime._key, runtime._state = 7799, "new-key", "starting"
    assert runtime._get_health(old) is None
    assert not calls


def test_port_allocation_failure_enters_backoff(tmp_path, monkeypatch):
    def fail():
        raise OSError("no ports")

    monkeypatch.setattr("voidcraft_world_bridge.services.free_loopback_port", fail)
    runtime = service(tmp_path)
    assert runtime.ensure_started() == "failed"
    assert "OSError" in runtime.status()["error"]


@pytest.fixture
def endpoint():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            self.send_response(302 if self.path == "/redirect" else 200)
            if self.path == "/redirect":
                self.send_header("Location", "/leak")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_port, seen
        finally:
            server.shutdown()
            thread.join(2)


def test_service_requests_ignore_proxy_environment(tmp_path, endpoint, monkeypatch):
    port, seen = endpoint
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"http": f"http://127.0.0.1:{port}"})
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)
    runtime = service(tmp_path)
    runtime._proc = SimpleNamespace(poll=lambda: None)
    runtime._port, runtime._key, runtime._state = port, "test-key", "ready"
    assert runtime.request("GET", "/work", None, 2) == (200, b"")
    # A proxy receives an absolute URL; only a direct request receives this path.
    assert seen == [("/work", "Bearer test-key")]


def test_service_does_not_forward_credentials_on_redirect(tmp_path, endpoint):
    port, seen = endpoint
    runtime = service(tmp_path)
    runtime._proc = SimpleNamespace(poll=lambda: None)
    runtime._port, runtime._key, runtime._state = port, "test-key", "ready"
    assert runtime.request("GET", "/redirect", None, 2) == (302, b"")
    assert seen == [("/redirect", "Bearer test-key")]


def test_service_path_cannot_change_destination(tmp_path):
    runtime = service(tmp_path)
    with pytest.raises(ValueError, match="absolute path"):
        runtime.request("GET", "@remote.example/work", None, 2)
