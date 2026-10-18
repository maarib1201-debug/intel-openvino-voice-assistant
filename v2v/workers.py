"""Bounded single-worker model gates. One shared model, many sessions."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Iterator


_END = object()


class BusyError(Exception):
    def __init__(self, retry_after_ms: int = 1500, reason: str = "queue-full"):
        super().__init__(reason)
        self.retry_after_ms = int(retry_after_ms)
        self.reason = reason


class Job:
    def __init__(
        self,
        session_id: str,
        fn: Callable[[], None],
        cancel: threading.Event,
        on_drop: Callable[[], None] | None = None,
    ):
        self.session_id = session_id
        self.fn = fn
        self.cancel = cancel
        self.queued_at = time.perf_counter()
        self.started_at = 0.0
        self.finished_at = 0.0
        self.wait_s = 0.0
        self.infer_s = 0.0
        self.skip = False
        self.error: BaseException | None = None
        self.on_drop = on_drop
        self._done = threading.Event()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout=timeout)


class ModelGate:
    """FIFO gate: one worker, bounded unique-session occupancy, max one live job per session."""

    def __init__(self, name: str, depth: int = 4):
        self.name = name
        self.depth = max(1, int(depth))
        self._cv = threading.Condition()
        self._pending: deque[Job] = deque()
        self._running: Job | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"v2v-{name}-gate")
        self._thread.start()

    def close(self) -> None:
        with self._cv:
            if self._stop.is_set():
                return
            self._stop.set()
            while self._pending:
                self._drop_unlocked(
                    self._pending.popleft(), RuntimeError(f"{self.name} gate closed")
                )
            self._cv.notify_all()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    def occupancy(self) -> set[str]:
        with self._cv:
            return self._occupancy_unlocked()

    def depth_now(self) -> int:
        with self._cv:
            n = len(self._pending)
            if self._running is not None:
                n += 1
            return n

    def _occupancy_unlocked(self) -> set[str]:
        ids = {job.session_id for job in self._pending}
        if self._running is not None:
            ids.add(self._running.session_id)
        return ids

    @staticmethod
    def _drop_unlocked(job: Job, error: BaseException | None = None) -> None:
        job.skip = True
        if error is not None:
            job.error = error
        if job.on_drop is not None:
            try:
                job.on_drop()
            except Exception as exc:
                job.error = job.error or exc
        job._done.set()

    def cancel_session(self, session_id: str) -> int:
        """Drop queued work for this session only. Running work sees its cancel Event."""
        dropped = 0
        with self._cv:
            kept: deque[Job] = deque()
            for job in self._pending:
                if job.session_id == session_id:
                    self._drop_unlocked(job)
                    dropped += 1
                else:
                    kept.append(job)
            self._pending = kept
            self._cv.notify_all()
        return dropped

    def submit(
        self,
        session_id: str,
        fn: Callable[[], None],
        cancel: threading.Event,
        on_drop: Callable[[], None] | None = None,
    ) -> Job:
        with self._cv:
            if self._stop.is_set():
                raise RuntimeError(f"{self.name} gate is closed")
            kept: deque[Job] = deque()
            for job in self._pending:
                if job.session_id == session_id:
                    self._drop_unlocked(job)
                else:
                    kept.append(job)
            self._pending = kept
            occupied = self._occupancy_unlocked()
            if session_id not in occupied and len(occupied) >= self.depth:
                retry = max(400, 400 * (len(occupied) + 1))
                raise BusyError(retry_after_ms=retry, reason=f"{self.name}-queue-full")
            job = Job(session_id, fn, cancel, on_drop=on_drop)
            self._pending.append(job)
            self._cv.notify()
            return job

    def run(self, session_id: str, fn: Callable[[], Any], cancel: threading.Event) -> tuple[Any, Job]:
        box: dict[str, Any] = {}

        def work() -> None:
            box["result"] = fn()

        job = self.submit(session_id, work, cancel)
        job.wait()
        if job.error is not None:
            raise job.error
        return box.get("result"), job

    def iterate(self, session_id: str, factory: Callable[[], Iterator], cancel: threading.Event) -> tuple[Iterator, Job]:
        out: Any = __import__("queue").Queue()

        def work() -> None:
            try:
                for item in factory():
                    if cancel.is_set():
                        break
                    out.put(item)
            finally:
                out.put(_END)

        job = self.submit(session_id, work, cancel, on_drop=lambda: out.put(_END))

        def gen() -> Iterator:
            while True:
                item = out.get()
                if item is _END:
                    break
                yield item
            job.wait()
            if job.error is not None:
                raise job.error

        return gen(), job

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while not self._pending and not self._stop.is_set():
                    self._cv.wait(timeout=0.2)
                if self._stop.is_set():
                    return
                job = self._pending.popleft()
                if job.skip:
                    continue
                self._running = job
            job.started_at = time.perf_counter()
            job.wait_s = job.started_at - job.queued_at
            try:
                if not job.skip and not job.cancel.is_set():
                    job.fn()
            except Exception as exc:
                job.error = exc
            finally:
                job.finished_at = time.perf_counter()
                job.infer_s = job.finished_at - job.started_at
                with self._cv:
                    if self._running is job:
                        self._running = None
                    self._cv.notify_all()
                job._done.set()
