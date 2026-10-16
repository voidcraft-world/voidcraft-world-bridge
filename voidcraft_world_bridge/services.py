"""Bridge-owned loopback children. Detached durable jobs use their own runner."""
from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

LOOPBACK = "127.0.0.1"


def find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    for candidate in ("~/.local/bin/uv", "/opt/homebrew/bin/uv", "/usr/local/bin/uv"):
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK, 0))
        return int(sock.getsockname()[1])


def tail(path: Path, lines: int = 6) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 8192))
            return "\n".join(stream.read().decode("utf-8", errors="replace").strip().splitlines()[-lines:])
    except OSError:
        return ""


class ManagedService:
    """Non-blocking startup, authenticated health probes, idle stop and backoff.

    command/environment are trusted local configuration, never request input.
    The child must bind loopback and enforce the generated bearer key. Health
    workers compare process identity before publishing state or stopping a child.
    """

    def __init__(self, command: Callable[[int], list[str] | None], *,
                 environment: Callable[[int, str], dict[str, str]], log_path: Path,
                 label: str = "service", cwd: Path | None = None,
                 ready: Callable[[dict], bool] = lambda health: health.get("status") == "ok",
                 idle_stop_seconds: float = 600, start_timeout_seconds: float = 600,
                 backoff_seconds: float = 30, poll_seconds: float = 0.5,
                 clock: Callable[[], float] = time.monotonic,
                 log: Callable[[str], None] = lambda message: None) -> None:
        self._command, self._environment = command, environment
        self._log_path, self._label, self._cwd = log_path, label, cwd
        self._ready = ready
        self._idle_stop_seconds, self._start_timeout = idle_stop_seconds, start_timeout_seconds
        self._backoff, self._poll_seconds = backoff_seconds, poll_seconds
        self._clock, self._log = clock, log
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._port: int | None = None
        self._key: str | None = None
        self._state = "stopped"
        self._error: str | None = None
        self._health: dict | None = None
        self._last_used, self._failed_at = clock(), float("-inf")
        self._active = 0
        self._closed = threading.Event()
        self._reaper_started = False

    def _current_state_locked(self) -> str:
        if self._state in ("starting", "ready") and self._proc is not None and self._proc.poll() is not None:
            self._state, self._failed_at = "failed", self._clock()
            self._error = f"{self._label} exited ({self._proc.returncode}): {tail(self._log_path)}"
            self._proc = None
        return self._state

    def state(self) -> str:
        with self._lock:
            return self._current_state_locked()

    def status(self) -> dict:
        with self._lock:
            state = self._current_state_locked()
            out: dict = {"state": state, "idle_stop_seconds": self._idle_stop_seconds}
            if self._error and state in ("failed", "unavailable"):
                out["error"] = self._error
            return out

    def ensure_started(self) -> str:
        with self._lock:
            state = self._current_state_locked()
            if self._closed.is_set():
                return "stopped"
            if state in ("starting", "ready", "installing", "not_installed"):
                if state == "ready":
                    self._last_used = self._clock()
                return state
            if state in ("failed", "unavailable") and self._clock() - self._failed_at < self._backoff:
                return state
            self._port, self._key = free_loopback_port(), secrets.token_urlsafe(24)
            try:
                argv = self._command(self._port)
                if argv is None:
                    self._state, self._failed_at = "unavailable", self._clock()
                    self._error = f"{self._label} command is unavailable"
                    return self._state
                env = {**os.environ, **self._environment(self._port, self._key)}
                env.pop("VIRTUAL_ENV", None)
                with self._log_path.open("w", encoding="utf-8") as output:
                    self._proc = subprocess.Popen(argv, env=env, cwd=self._cwd, stdout=output,
                                                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                                  start_new_session=True)
            except (OSError, ValueError) as exc:
                self._state, self._failed_at = "failed", self._clock()
                self._error = f"{self._label} could not start: {type(exc).__name__}"
                return self._state
            self._state, self._error, self._health = "starting", None, None
            self._last_used = self._clock()
            threading.Thread(target=self._await_health, args=(self._proc,), daemon=True,
                             name=f"{self._label}-health").start()
            if not self._reaper_started:
                self._reaper_started = True
                threading.Thread(target=self._reap_forever, daemon=True, name=f"{self._label}-idle").start()
            return self._state

    def _get_health(self) -> dict | None:
        reply = self.request("GET", "/health", None, 2)
        if reply is None or reply[0] != 200:
            return None
        try:
            health = json.loads(reply[1])
            return health if isinstance(health, dict) else None
        except ValueError:
            return None

    def _await_health(self, proc: subprocess.Popen) -> None:
        deadline = self._clock() + self._start_timeout
        while not self._closed.is_set() and self._clock() < deadline:
            with self._lock:
                if self._proc is not proc or proc.poll() is not None:
                    return
            try:
                health = self._get_health()
                ready = health is not None and self._ready(health)
            except Exception:  # noqa: BLE001 — a probe error is retried until the startup deadline
                ready, health = False, None
            if ready:
                with self._lock:
                    if self._proc is proc:
                        self._state, self._health = "ready", health
                        self._last_used = self._clock()
                return
            self._closed.wait(self._poll_seconds)
        with self._lock:
            if self._proc is not proc or self._closed.is_set():
                return
            self._error = f"{self._label} did not become healthy in {self._start_timeout}s"
        self.stop(reason="start timed out", keep_error=True, expected_proc=proc)

    def stop(self, reason: str = "requested", keep_error: bool = False,
             *, expected_proc: subprocess.Popen | None = None, only_if_idle: bool = False) -> bool:
        with self._lock:
            if expected_proc is not None and self._proc is not expected_proc:
                return False
            if only_if_idle and (self._active or self._state != "ready"
                                 or self._clock() - self._last_used <= self._idle_stop_seconds):
                return False
            proc, self._proc = self._proc, None
            self._state, self._health = ("failed" if keep_error else "stopped"), None
            if keep_error:
                self._failed_at = self._clock()
            else:
                self._error = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            self._log(f"[{self._label}] stopped ({reason})")
        return True

    def _reap_forever(self) -> None:
        while not self._closed.wait(30):
            self.reap_if_idle()

    def reap_if_idle(self) -> bool:
        with self._lock:
            idle = (self._state == "ready" and self._active == 0
                    and self._clock() - self._last_used > self._idle_stop_seconds)
            proc = self._proc
        if idle and proc is not None:
            return self.stop(reason="idle", expected_proc=proc, only_if_idle=True)
        return False

    def request(self, method: str, path: str, body: dict | None, timeout: float) -> tuple[int, bytes] | None:
        with self._lock:
            port, key = self._port, self._key
            if port is None or key is None:
                return None
            self._active += 1
        try:
            request = urllib.request.Request(
                f"http://{LOOPBACK}:{port}{path}", method=method,
                data=None if body is None else json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return response.status, response.read()
            except urllib.error.HTTPError as exc:
                return exc.code, exc.read()
            except (OSError, ValueError):
                return None
        finally:
            with self._lock:
                self._active -= 1
                self._last_used = self._clock()

    def close(self) -> None:
        self._closed.set()
        self.stop(reason="bridge shutdown")
