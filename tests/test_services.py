from __future__ import annotations

from types import SimpleNamespace

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
    monkeypatch.setattr(runtime, "request", lambda *args: (200, b"[]"))
    assert runtime._get_health() is None


def test_stop_terminates_only_owned_child(tmp_path):
    calls = []
    runtime = service(tmp_path)
    runtime._proc = SimpleNamespace(poll=lambda: None, terminate=lambda: calls.append("terminate"),
                                    wait=lambda timeout: calls.append(timeout))
    runtime.close()
    runtime.close()
    assert calls == ["terminate", 10]
