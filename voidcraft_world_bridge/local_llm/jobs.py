"""A bounded, single-worker job queue — why generation is a job and not a request.

── WHY A JOB ─────────────────────────────────────────────────────────────────
The host rule is "never block long": plugin handlers run on the bridge's
request threads, which serve every other route too. One answer from a local
model is 5–60 s, and the FIRST one after a cold start also pages the model in
from disk. So `POST /chat` enqueues and returns 202 at once, and the page polls
`GET /job`.

── WHILE IT RUNS ─────────────────────────────────────────────────────────────
Polling alone can say only "running", for up to a minute. A store built with
`reports_progress=True` passes its runner a second argument, `report`, and a
running job's `progress` is whatever the runner reported last — for the chat
runner, the answer so far (`progress.py` owns that shape and its throttle).
The worker writes it and request threads read it, so each report is copied in
under the store's lock. `progress` is null while queued, for a runner that
never reports, and once the job is done or failed: `result` or `error` is the
answer then. A one-argument runner works as before.

── WHY ONE WORKER ────────────────────────────────────────────────────────────
Two concurrent generations on one Mac do not finish sooner; they split the
same memory bandwidth, and two DIFFERENT models loaded at once can exceed the
GPU budget and push both onto the CPU. Serial is the fast order here.

── ONE STORE PER MACHINE ─────────────────────────────────────────────────────
"One worker" is per model server. Another machine (`hosts.py`) has its own
memory bandwidth, so the plugin gives each host its own store, and the job
records which `host` decoded it.

── WHY BOUNDED, TWICE ────────────────────────────────────────────────────────
`MAX_PENDING` refuses new work while four jobs wait, because the fifth
question would not start for minutes and the page that asked it has usually
moved on. `MAX_KEPT` forgets the oldest FINISHED jobs, because nothing ever
deletes a job otherwise and the bridge runs for weeks.
"""
from __future__ import annotations

import copy
import functools
import queue
import secrets
import threading
import time
from collections.abc import Callable
from typing import cast

MAX_PENDING = 4
MAX_KEPT = 32

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"

Runner = Callable[[dict], dict]
"""Takes the job's `request`, returns the result dict or raises JobFailure."""

Report = Callable[[dict], None]
"""A running job's progress sink. Each call replaces the job's `progress` with
a copy of the JSON-able dict it is given, until the job finishes."""

ProgressRunner = Callable[[dict, Report], dict]
"""A Runner that also takes the job's `report` (`reports_progress=True`)."""


class JobFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class QueueFull(Exception):
    pass


class JobStore:
    def __init__(self, runner: Runner | ProgressRunner, *, reports_progress: bool = False,
                 clock: Callable[[], float] = time.time,
                 start_worker: bool = True, max_pending: int = MAX_PENDING,
                 max_kept: int = MAX_KEPT, host: str = "local") -> None:
        if max_pending < 1 or max_kept < 1:
            raise ValueError("job limits must be positive")
        self._max_pending, self._max_kept = max_pending, max_kept
        self._closed = False
        # One calling shape inside: a plain runner is wrapped to ignore its sink.
        self._runner: ProgressRunner = (cast(ProgressRunner, runner) if reports_progress
                                        else _ignoring_report(cast(Runner, runner)))
        self.host = host
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._order: list[str] = []
        self._pending: queue.Queue[str | None] = queue.Queue()
        if start_worker:
            threading.Thread(target=self._work_forever, name=f"local-llm-worker-{host}", daemon=True).start()

    # -- public ---------------------------------------------------------------
    def submit(self, request: dict) -> dict:
        with self._lock:
            waiting = sum(1 for j in self._jobs.values() if j["state"] == QUEUED)
            if self._closed or waiting >= self._max_pending:
                raise QueueFull()
            job_id = secrets.token_hex(8)
            job = {"id": job_id, "state": QUEUED, "created_at": self._clock(),
                   "started_at": None, "finished_at": None,
                   "model": request.get("model"), "result": None, "error": None,
                   "progress": None, "request": request}
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._forget_oldest_finished()
            snapshot = self._public(job)
            self._pending.put(job_id)
        return snapshot

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._public(job) if job else None

    def counts(self) -> dict:
        with self._lock:
            states = [j["state"] for j in self._jobs.values()]
        return {"queued": states.count(QUEUED), "running": states.count(RUNNING)}

    def load(self) -> int:
        """Queued + running: how far back a new question would start."""
        counts = self.counts()
        return counts["queued"] + counts["running"]

    def run_next(self) -> bool:
        """Run one queued job on THIS thread. The worker loop calls it; tests call it directly."""
        try:
            job_id = self._pending.get_nowait()
        except queue.Empty:
            return False
        if job_id is None:
            return False
        self._run(job_id)
        return True

    def close(self) -> None:
        """Drain accepted work and stop the worker; reject subsequent submissions."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._pending.put(None)

    # -- internals ------------------------------------------------------------
    def _work_forever(self) -> None:
        while True:
            job_id = self._pending.get()
            if job_id is None:
                return
            self._run(job_id)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["state"] = RUNNING
            job["started_at"] = self._clock()
            request = dict(job["request"])
        try:
            result = self._runner(request, functools.partial(self._report, job))
            # Timings describe the job, not the answer: lifted out so `result` stays the adapter's shape.
            timings = result.pop("timings", None)
            outcome = {"state": DONE, "result": result, "timings": timings,
                       "model": result.get("model") or request.get("model")}
        except JobFailure as failure:
            outcome = {"state": FAILED, "error": {"code": failure.code, "message": failure.message}}
        except Exception as err:  # noqa: BLE001 — a runner bug must fail ONE job, never kill the worker
            outcome = {"state": FAILED, "error": {"code": "internal", "message": str(err)[:300]}}
        with self._lock:
            job.update(outcome)
            job["finished_at"] = self._clock()
            job["progress"] = None  # the answer is `result` now; the partial text goes with the prompt
            job["request"] = {}  # the prompt is the user's content; do not keep it past the answer

    def _report(self, job: dict, progress: dict) -> None:
        """The runner's `report`. Copied in, so a runner that keeps mutating its
        dict cannot change what a reader is serializing; ignored once finished."""
        snapshot = copy.deepcopy(progress)
        with self._lock:
            if job["state"] == RUNNING:
                job["progress"] = snapshot

    def _forget_oldest_finished(self) -> None:
        while len(self._order) > self._max_kept:
            victim = next((jid for jid in self._order if self._jobs[jid]["state"] in (DONE, FAILED)), None)
            if victim is None:
                return
            self._order.remove(victim)
            del self._jobs[victim]

    def _public(self, job: dict) -> dict:
        started = job["started_at"]
        finished = job["finished_at"]
        now = self._clock()
        elapsed = None
        if started is not None:
            elapsed = int(((finished if finished is not None else now) - started) * 1000)
        return {
            "id": job["id"],
            "state": job["state"],
            "model": job["model"],
            "queued_ms": int(((started if started is not None else now) - job["created_at"]) * 1000),
            "elapsed_ms": elapsed,
            "result": job["result"],
            "error": job["error"],
            "host": self.host,
            "timings": job.get("timings"),
            # Replaced on each report, never mutated, so it is safe to serialize outside the lock.
            "progress": job["progress"] if job["state"] == RUNNING else None,
        }


def _ignoring_report(runner: Runner) -> ProgressRunner:
    return lambda request, _report: runner(request)
