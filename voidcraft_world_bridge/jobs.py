"""Background work shared by plugins: bounded queues and a reject-while-busy slot."""
from __future__ import annotations

import threading
from collections.abc import Callable
from copy import deepcopy

# Compatibility: the LLM's wire shape stays owned by its original implementation.
from voidcraft_world_bridge.local_llm.jobs import JobFailure, JobStore, QueueFull, Report

__all__ = ["BackgroundJob", "JobFailure", "JobStore", "QueueFull", "Report"]


class BackgroundJob:
    """One non-queued job. The runner owns its result shape and subprocess timeout.

    Accepted work finishes on its daemon thread. close() prevents new work; it
    cannot cancel an arbitrary callable. Use busy_reasons() before a restart.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: str | None = None
        self._last_run: dict | None = None
        self._closed = False

    def start(self, name: str, runner: Callable[[], dict]) -> bool:
        with self._lock:
            if self._closed or self._running is not None:
                return False
            self._running = name
        try:
            threading.Thread(target=self._run, args=(runner,), name=f"job-{name}", daemon=True).start()
        except Exception:
            with self._lock:
                self._running = None
            raise
        return True

    def _run(self, runner: Callable[[], dict]) -> None:
        try:
            result = runner()
            if not isinstance(result, dict):
                raise TypeError("job runner must return a dict")
            result = deepcopy(result)
        except Exception as exc:  # noqa: BLE001 — one failed job must release its slot
            result = {"ok": False, "error": type(exc).__name__}
        with self._lock:
            self._last_run, self._running = result, None

    def snapshot(self) -> dict:
        with self._lock:
            return {"running": self._running, "last_run": deepcopy(self._last_run)}

    def busy_reasons(self) -> list[str]:
        with self._lock:
            return [self._running] if self._running is not None else []

    def close(self) -> None:
        with self._lock:
            self._closed = True
