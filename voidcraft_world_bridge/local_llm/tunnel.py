"""An SSH port forward to another machine's model server — see `hosts.py` for why SSH.

    ssh -o BatchMode=yes … -L 127.0.0.1:<free port>:127.0.0.1:<remote port> -- <alias> <keep-awake>

── WHAT EACH PIECE IS FOR ────────────────────────────────────────────────────
- `BatchMode=yes`: never prompt. The bridge has no terminal; a host that needs
  a password or an unknown host key fails fast instead of hanging. Set it up
  once by hand (`ssh <alias>`), and the tunnel works from then on.
- `ExitOnForwardFailure=yes`: a forward that could not bind is a dead tunnel,
  not a live SSH session forwarding nothing.
- `ServerAlive*`: a machine that went to sleep or left the network is noticed
  in ~45 s, and the tunnel is restarted on the next use rather than trusted.
- The forward binds 127.0.0.1 on BOTH ends: this machine's loopback, and the
  other machine's. Neither listens on the network.
- `<keep-awake>`: the remote command the session holds open. It keeps the other
  machine from idle-sleeping exactly while the tunnel is up, and no longer —
  `caffeinate -i` on macOS, `systemd-inhibit` on Linux, found once by `uname -s`.
- The command it holds is `cat`, reading ssh's stdin — a pipe only this bridge
  holds open. If the bridge dies by any means (a SIGKILL skips every cleanup),
  the pipe closes, `cat` reads EOF and exits, and the session ends with it. A
  tunnel can never outlive its bridge and keep the other machine awake forever.

── LIFECYCLE ─────────────────────────────────────────────────────────────────
Started lazily (the first `/status` or question that needs the host), on a
background thread, because the first connect can take seconds and `/status`
runs on a request thread. A failed start backs off before trying again. An
unused tunnel stops after `idle_stop_seconds`, so the other machine is free to
sleep once nobody is asking it anything.
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Callable

from voidcraft_world_bridge.services import LOOPBACK, free_loopback_port

NONE, STARTING, UP, DOWN = "none", "starting", "up", "down"
"""Tunnel states as `/status` reports them. NONE is this machine: no tunnel."""

CONNECT_TIMEOUT_SECONDS = 10
READY_TIMEOUT_SECONDS = 20.0
IDLE_STOP_SECONDS = 30 * 60
BACKOFF_SECONDS = (5.0, 15.0, 60.0)

KEEP_AWAKE = {
    "Darwin": ["caffeinate", "-i", "cat"],
    "Linux": ["systemd-inhibit", "--what=sleep:idle", "--why=voidcraft-local-llm", "cat"],
}
HOLD_OPEN = ["cat"]
"""Any other system: hold the session open without keeping anything awake."""

SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
    "-o", f"ConnectTimeout={CONNECT_TIMEOUT_SECONDS}",
)


def keep_awake_command(system: str) -> list[str]:
    return list(KEEP_AWAKE.get(system.strip(), HOLD_OPEN))


def port_open(port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((LOOPBACK, port), timeout=timeout):
            return True
    except OSError:
        return False


class SshTunnel:
    """One forward to one alias. Thread-safe; every public method returns at once
    except `wait_up`, which the job worker calls before a question."""

    def __init__(self, alias: str, remote_port: int, *, ssh: str | None = None,
                 idle_stop_seconds: float = IDLE_STOP_SECONDS,
                 ready_timeout_seconds: float = READY_TIMEOUT_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.alias = alias
        self.remote_port = remote_port
        self._ssh = ssh
        self._idle_stop = idle_stop_seconds
        self._ready_timeout = ready_timeout_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._proc: subprocess.Popen | None = None
        self._port: int | None = None
        self._state = DOWN
        self._error: str | None = None
        self._system: str | None = None
        self._failures = 0
        self._retry_at = float("-inf")
        self._last_used = clock()
        self._closed = False

    # -- public ---------------------------------------------------------------
    def snapshot(self) -> tuple[str, int | None, str | None]:
        """(state, local port when UP, last error). Never starts anything."""
        with self._lock:
            self._notice_exit_locked()
            return self._state, (self._port if self._state == UP else None), self._error

    def ensure(self) -> tuple[str, int | None, str | None]:
        """Mark the tunnel used and start it if it is down (and not backing off)."""
        with self._lock:
            self._notice_exit_locked()
            self._last_used = self._clock()
            if self._state == DOWN and not self._closed and self._clock() >= self._retry_at:
                self._state, self._error = STARTING, None
                threading.Thread(target=self._start, name=f"tunnel-{self.alias}", daemon=True).start()
        self.reap_if_idle()
        return self.snapshot()

    def wait_up(self, timeout: float | None = None) -> int | None:
        """The local port once the tunnel is UP, or None if it did not come up in time."""
        self.ensure()
        deadline = self._clock() + (self._ready_timeout if timeout is None else timeout)
        with self._lock:
            while True:
                self._notice_exit_locked()
                if self._state == UP:
                    self._last_used = self._clock()
                    return self._port
                remaining = deadline - self._clock()
                if self._state == DOWN or remaining <= 0:
                    return None
                self._changed.wait(min(remaining, 0.5))

    def reap_if_idle(self) -> bool:
        with self._lock:
            idle = self._state == UP and self._clock() - self._last_used > self._idle_stop
        if idle:
            self._stop("idle")
        return idle

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._stop("closed")

    # -- internals ------------------------------------------------------------
    def _ssh_binary(self) -> str:
        return self._ssh or shutil.which("ssh") or "ssh"

    def _notice_exit_locked(self) -> None:
        """A tunnel whose ssh exited (the host slept, the network dropped) is DOWN."""
        if self._state == UP and self._proc is not None and self._proc.poll() is not None:
            self._fail_locked(self._exit_reason(self._proc))

    def _fail_locked(self, error: str) -> None:
        self._state, self._error, self._proc, self._port = DOWN, error, None, None
        self._failures += 1
        backoff = BACKOFF_SECONDS[min(self._failures, len(BACKOFF_SECONDS)) - 1]
        self._retry_at = self._clock() + backoff
        self._changed.notify_all()

    @staticmethod
    def _exit_reason(proc: subprocess.Popen) -> str:
        detail = ""
        if proc.stderr is not None:
            try:
                lines = [ln for ln in proc.stderr.read().decode("utf-8", "replace").splitlines()
                         if ln.strip() and not ln.startswith("**")]
                detail = lines[-1].strip()[:200] if lines else ""
            except (OSError, ValueError):
                detail = ""
        return f"ssh exited ({proc.returncode})" + (f": {detail}" if detail else "")

    def _detect_system(self) -> str:
        """`uname -s` on the other machine, once per tunnel object."""
        if self._system is None:
            done = subprocess.run([self._ssh_binary(), *SSH_OPTIONS, "--", self.alias, "uname", "-s"],
                                  capture_output=True, timeout=CONNECT_TIMEOUT_SECONDS + 5, check=False,
                                  stdin=subprocess.DEVNULL)
            if done.returncode != 0:
                lines = [ln for ln in done.stderr.decode("utf-8", "replace").splitlines()
                         if ln.strip() and not ln.startswith("**")]
                raise OSError(lines[-1].strip()[:200] if lines else f"ssh exited ({done.returncode})")
            self._system = done.stdout.decode("utf-8", "replace").strip()
        return self._system

    def _start(self) -> None:
        try:
            system = self._detect_system()
            port = free_loopback_port()
            argv = [self._ssh_binary(), *SSH_OPTIONS,
                    "-L", f"{LOOPBACK}:{port}:{LOOPBACK}:{self.remote_port}",
                    "--", self.alias, *keep_awake_command(system)]
            # stdin=PIPE is the lifeline (module note): never written, closed only by our exit.
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, start_new_session=True)
        except (OSError, ValueError, subprocess.SubprocessError) as err:
            with self._lock:
                self._fail_locked(f"ssh could not connect: {err}")
            return
        deadline = self._clock() + self._ready_timeout
        while self._clock() < deadline:
            if proc.poll() is not None:
                with self._lock:
                    self._fail_locked(self._exit_reason(proc))
                return
            if port_open(port):
                with self._lock:
                    if self._closed:
                        break
                    self._proc, self._port, self._state = proc, port, UP
                    self._error, self._failures = None, 0
                    self._last_used = self._clock()
                    self._changed.notify_all()
                return
            time.sleep(0.1)
        _terminate(proc)
        with self._lock:
            self._fail_locked("closed" if self._closed else f"the forward did not open in {self._ready_timeout:.0f}s")

    def _stop(self, reason: str) -> None:
        with self._lock:
            proc, self._proc, self._port = self._proc, None, None
            if self._state != DOWN:
                self._state = DOWN
                self._error = None if reason == "idle" else self._error
            self._changed.notify_all()
        if proc is not None:
            _terminate(proc)


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    for stream in (proc.stdin, proc.stderr):
        if stream is not None:
            stream.close()
