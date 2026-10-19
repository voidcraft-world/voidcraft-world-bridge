"""The real `SshTunnel`, against a fake `ssh` that forwards nothing but opens the port.

The fake records its argv, answers `uname -s`, and for a forward binds the
local half of `-L` and holds it open until terminated — what ssh does, minus
the network. `FAKE_SSH_MODE` makes it fail like a host that refuses the key.
"""
from __future__ import annotations

import json
import socket
import sys
import time

import pytest

from voidcraft_world_bridge.local_llm import tunnel as tunnel_mod
from voidcraft_world_bridge.local_llm.tunnel import DOWN, UP, SshTunnel, keep_awake_command

FAKE_SSH = """\
import json, os, signal, socket, sys, time
argv = sys.argv[1:]
log = os.environ["FAKE_SSH_LOG"]
with open(log, "a") as stream:
    stream.write(json.dumps(argv) + "\\n")
mode = os.environ.get("FAKE_SSH_MODE", "ok")
if mode == "denied":
    sys.stderr.write("** WARNING: connection is not using a post-quantum key exchange algorithm.\\n")
    sys.stderr.write("user@host: Permission denied (publickey).\\n")
    sys.exit(255)
if argv[-2:] == ["uname", "-s"]:
    print(os.environ.get("FAKE_SSH_SYSTEM", "Linux"))
    sys.exit(0)
spec = argv[argv.index("-L") + 1]
host, port = spec.split(":")[0], int(spec.split(":")[1])
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((host, port))
server.listen(4)
if mode == "dies":
    time.sleep(0.3)
    sys.stderr.write("client_loop: send disconnect: Broken pipe\\n")
    sys.exit(255)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
while True:
    time.sleep(0.1)
"""


@pytest.fixture
def fake_ssh(tmp_path, monkeypatch):
    script = tmp_path / "ssh"
    script.write_text(f"#!{sys.executable}\n{FAKE_SSH}")
    script.chmod(0o755)
    log = tmp_path / "argv.log"
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))

    def calls() -> list[list[str]]:
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    return str(script), calls


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_keep_awake_follows_the_other_machines_system():
    assert keep_awake_command("Darwin\n") == ["caffeinate", "-i"]
    assert keep_awake_command("Linux")[0] == "systemd-inhibit"
    assert keep_awake_command("FreeBSD") == ["sleep", "infinity"]


def test_the_tunnel_opens_a_loopback_forward_and_holds_the_host_awake(fake_ssh):
    ssh, calls = fake_ssh
    tunnel = SshTunnel("studio", 11434, ssh=ssh)
    try:
        port = tunnel.wait_up(timeout=10)
        assert port is not None
        assert tunnel.snapshot() == (UP, port, None)
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
        detect, forward = calls()
        assert detect[-4:] == ["--", "studio", "uname", "-s"]
        assert "BatchMode=yes" in forward and "ExitOnForwardFailure=yes" in forward
        assert forward[forward.index("-L") + 1] == f"127.0.0.1:{port}:127.0.0.1:11434"
        separator = forward.index("--")
        assert forward[separator + 1] == "studio"
        assert forward[separator + 2:] == keep_awake_command("Linux")
        assert all(not arg.startswith("studio") for arg in forward[:separator]), "the alias only after --"
    finally:
        tunnel.close()
    assert tunnel.snapshot()[0] == DOWN


def test_the_system_is_detected_once_per_tunnel(fake_ssh, monkeypatch):
    ssh, calls = fake_ssh
    monkeypatch.setenv("FAKE_SSH_SYSTEM", "Darwin")
    tunnel = SshTunnel("studio", 11434, ssh=ssh, idle_stop_seconds=0)
    try:
        assert tunnel.wait_up(timeout=10)
        time.sleep(0.01)
        assert tunnel.reap_if_idle() is True
        assert tunnel.wait_up(timeout=10)
        unames = [argv for argv in calls() if argv[-2:] == ["uname", "-s"]]
        assert len(unames) == 1
        assert calls()[-1][-2:] == ["caffeinate", "-i"]
    finally:
        tunnel.close()


def test_a_refused_key_is_down_with_ssh_s_own_reason_and_backs_off(fake_ssh, monkeypatch):
    ssh, calls = fake_ssh
    monkeypatch.setenv("FAKE_SSH_MODE", "denied")
    tunnel = SshTunnel("studio", 11434, ssh=ssh)
    assert tunnel.wait_up(timeout=10) is None
    state, port, error = tunnel.snapshot()
    assert (state, port) == (DOWN, None)
    assert "Permission denied (publickey)" in error and "post-quantum" not in error
    before = len(calls())
    tunnel.ensure()
    time.sleep(0.2)
    assert len(calls()) == before, "a failed host is not retried inside its backoff"


def test_a_tunnel_whose_ssh_exits_is_noticed(fake_ssh, monkeypatch):
    ssh, _ = fake_ssh
    monkeypatch.setenv("FAKE_SSH_MODE", "dies")
    monkeypatch.setattr(tunnel_mod, "BACKOFF_SECONDS", (60.0,))
    tunnel = SshTunnel("studio", 11434, ssh=ssh)
    assert tunnel.wait_up(timeout=10) is not None
    assert wait_for(lambda: tunnel.snapshot()[0] == DOWN)
    assert "Broken pipe" in tunnel.snapshot()[2]
    tunnel.close()


def test_an_ssh_that_is_not_installed_is_down_not_a_crash(tmp_path):
    tunnel = SshTunnel("studio", 11434, ssh=str(tmp_path / "no-such-ssh"))
    assert tunnel.wait_up(timeout=5) is None
    assert tunnel.snapshot()[2].startswith("ssh could not connect")
