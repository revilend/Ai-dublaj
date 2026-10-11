"""In-memory asynchronous background task manager.

Long-running work (voice cloning, dubbing renders) must never block the request
loop. Instead, an endpoint creates a task, schedules a coroutine and returns
``202 Accepted`` immediately. The caller then either polls
``GET /api/tasks/{id}/status`` or streams ``/ws/tasks/{id}``.

Design notes
------------
* Records live in an ``OrderedDict`` guarded by a re-entrant ``threading.Lock``
  because progress is reported from worker threads created by
  ``asyncio.to_thread`` / ``run_in_executor``.
* Subscribers are plain ``asyncio.Queue`` objects; publishing from a worker
  thread is routed back onto the event loop with ``call_soon_threadsafe``.
* Tasks are *cooperative*: blocking engine calls cannot be interrupted, so
  cancellation is observed at the next ``ctx.raise_if_cancelled()`` checkpoint.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import AppError, TaskNotFoundError
from app.core.logging import get_logger
from app.schemas.task import TaskState, TaskStatusResponse

logger = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TaskCancelled(Exception):
    """Raised inside a task body when cancellation has been requested."""


@dataclass
class TaskRecord:
    """Mutable state for one background job."""

    id: str
    name: str
    kind: str = "generic"
    status: TaskState = TaskState.PENDING
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    started_at: datetime | None = None
    updated_at: datetime = field(default_factory=_utcnow)
    finished_at: datetime | None = None
    cancel_requested: bool = False


class TaskContext:
    """Handle passed to a task coroutine for progress reporting."""

    __slots__ = ("_manager", "_task_id")

    def __init__(self, manager: "TaskManager", task_id: str) -> None:
        self._manager = manager
        self._task_id = task_id

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def record(self) -> TaskRecord:
        return self._manager.get(self._task_id)

    @property
    def cancelled(self) -> bool:
        return self._manager.get(self._task_id).cancel_requested

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise TaskCancelled()

    def report(
        self,
        progress: float | None = None,
        message: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._manager.update(
            self._task_id, progress=progress, message=message, metadata=metadata
        )

    def complete(self, result: dict[str, Any] | None = None, message: str = "Completed") -> None:
        self._manager.update(
            self._task_id, progress=100.0, status=TaskState.COMPLETED,
            result=result, message=message,
        )

    def fail(self, error: str, message: str | None = None) -> None:
        self._manager.update(
            self._task_id,
            status=TaskState.FAILED,
            error=error,
            message=message or "Task failed",
        )


TaskBody = Callable[[TaskContext], Awaitable[dict[str, Any] | None]]


class TaskManager:
    """Thread-safe, in-memory registry + scheduler for background work."""

    def __init__(
        self,
        *,
        retention_seconds: int | None = None,
        max_history: int | None = None,
    ) -> None:
        settings = get_settings()
        self._retention_seconds = (
            settings.task_retention_seconds if retention_seconds is None else retention_seconds
        )
        self._max_history = settings.task_max_history if max_history is None else max_history

        self._records: "OrderedDict[str, TaskRecord]" = OrderedDict()
        self._subscribers: dict[str, set[asyncio.Queue[TaskStatusResponse]]] = {}
        self._running: dict[str, asyncio.Task[None]] = {}
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------ #
    # Loop binding
    # ------------------------------------------------------------------ #
    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Remember the running loop so worker threads can publish events."""
        with self._lock:
            self._loop = loop or asyncio.get_event_loop()

    # ------------------------------------------------------------------ #
    # Registry
    # ------------------------------------------------------------------ #
    def create(
        self,
        name: str,
        *,
        kind: str = "generic",
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord:
        record = TaskRecord(
            id=uuid.uuid4().hex,
            name=name,
            kind=kind,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._records[record.id] = record
            self._evict_locked()
        logger.info("Task %s created (%s)", record.id, name)
        return record

    def find(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._records.get(task_id)

    def get(self, task_id: str) -> TaskRecord:
        record = self.find(task_id)
        if record is None:
            raise TaskNotFoundError(
                f"Task '{task_id}' does not exist or has expired.",
                details={"task_id": task_id},
            )
        return record

    def list(
        self,
        *,
        limit: int = 50,
        status: TaskState | None = None,
        kind: str | None = None,
    ) -> list[TaskRecord]:
        with self._lock:
            records = list(self._records.values())
        records.sort(key=lambda item: item.created_at, reverse=True)
        if status is not None:
            records = [item for item in records if item.status == status]
        if kind is not None:
            records = [item for item in records if item.kind == kind]
        return records[: max(1, limit)]

    # ------------------------------------------------------------------ #
    # State transitions
    # ------------------------------------------------------------------ #
    def update(
        self,
        task_id: str,
        *,
        progress: float | None = None,
        message: str | None = None,
        status: TaskState | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord:
        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                raise TaskNotFoundError(
                    f"Task '{task_id}' does not exist or has expired.",
                    details={"task_id": task_id},
                )

            if progress is not None:
                record.progress = max(0.0, min(100.0, float(progress)))
            if message is not None:
                record.message = message
            if result is not None:
                record.result = result
            if error is not None:
                record.error = error
            if metadata:
                record.metadata.update(metadata)

            if status is not None and status != record.status:
                record.status = status
                if status is TaskState.RUNNING and record.started_at is None:
                    record.started_at = _utcnow()
                if status.is_terminal:
                    record.finished_at = _utcnow()

            record.updated_at = _utcnow()
            snapshot = TaskStatusResponse.model_validate(record)

        self._publish(task_id, snapshot)
        return record

    def cancel(self, task_id: str) -> bool:
        """Request cancellation. Returns ``False`` if already terminal."""
        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                raise TaskNotFoundError(f"Task '{task_id}' does not exist.")
            if record.status.is_terminal:
                return False
            record.cancel_requested = True
            running = self._running.get(task_id)
            loop = self._loop

        if running is not None and not running.done():
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(running.cancel)
            else:  # pragma: no cover - defensive
                running.cancel()
        else:
            self.update(task_id, status=TaskState.CANCELLED, message="Cancelled")
        return True

    def cleanup(self, *, now: datetime | None = None) -> int:
        """Drop finished records older than the retention window."""
        if self._retention_seconds <= 0:
            return 0
        reference = now or _utcnow()
        removed = 0
        with self._lock:
            for task_id, record in list(self._records.items()):
                if not record.status.is_terminal or record.finished_at is None:
                    continue
                finished = record.finished_at
                if finished.tzinfo is None:  # pragma: no cover - sqlite round-trip
                    finished = finished.replace(tzinfo=timezone.utc)
                if (reference - finished).total_seconds() >= self._retention_seconds:
                    self._records.pop(task_id, None)
                    removed += 1
        if removed:
            logger.debug("Cleaned up %d expired task(s)", removed)
        return removed

    def stats(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {state.value: 0 for state in TaskState}
            for record in self._records.values():
                counts[record.status.value] += 1
            counts["total"] = len(self._records)
            counts["running_handles"] = len(self._running)
        return counts

    # ------------------------------------------------------------------ #
    # Scheduling
    # ------------------------------------------------------------------ #
    def start(self, record: TaskRecord, body: TaskBody) -> asyncio.Task[None]:
        """Schedule ``body`` and track it. Must be called inside the event loop."""
        loop = asyncio.get_running_loop()
        self.bind_loop(loop)

        async def runner() -> None:
            ctx = TaskContext(self, record.id)
            try:
                result = await body(ctx)
                if self.find(record.id) is None:  # pragma: no cover
                    return
                current = self.get(record.id)
                if not current.status.is_terminal:
                    ctx.complete(result)
            except TaskCancelled:
                self._safe_finish(record.id, TaskState.CANCELLED, error=None, message="Cancelled")
            except asyncio.CancelledError:
                self._safe_finish(record.id, TaskState.CANCELLED, error=None, message="Cancelled")
                raise
            except AppError as exc:
                logger.warning("Task %s failed: %s", record.id, exc.message)
                self._safe_finish(
                    record.id, TaskState.FAILED, error=f"{exc.code}: {exc.message}",
                    message=exc.message,
                )
            except Exception as exc:  # noqa: BLE001 - surface unexpected failures
                logger.exception("Task %s crashed", record.id)
                self._safe_finish(
                    record.id, TaskState.FAILED, error=str(exc), message="Unexpected failure"
                )
            finally:
                with self._lock:
                    self._running.pop(record.id, None)

        task = loop.create_task(runner(), name=f"task:{record.name}:{record.id[:8]}")
        with self._lock:
            self._running[record.id] = task
        self.update(record.id, status=TaskState.RUNNING, message="Starting…", progress=0.0)
        return task

    def _safe_finish(
        self,
        task_id: str,
        status: TaskState,
        *,
        error: str | None,
        message: str,
    ) -> None:
        try:
            self.update(task_id, status=status, error=error, message=message)
        except TaskNotFoundError:  # pragma: no cover - expired mid-flight
            logger.warning("Task %s disappeared before it could finish", task_id)

    # ------------------------------------------------------------------ #
    # Pub/sub
    # ------------------------------------------------------------------ #
    def _publish(self, task_id: str, snapshot: TaskStatusResponse) -> None:
        with self._lock:
            queues = list(self._subscribers.get(task_id, ()))
            loop = self._loop
        for queue in queues:
            if loop is not None and loop.is_running():
                try:
                    loop.call_soon_threadsafe(self._safe_put, queue, snapshot)
                except RuntimeError:  # pragma: no cover - loop closed
                    pass
            else:  # pragma: no cover - sync tests
                self._safe_put(queue, snapshot)

    @staticmethod
    def _safe_put(queue: asyncio.Queue[TaskStatusResponse], snapshot: TaskStatusResponse) -> None:
        try:
            queue.put_nowait(snapshot)
        except asyncio.QueueFull:  # pragma: no cover - slow consumer
            pass

    async def subscribe(
        self, task_id: str, *, heartbeat: float = 15.0
    ) -> AsyncIterator[TaskStatusResponse]:
        """Yield status snapshots until the task reaches a terminal state."""
        if self.find(task_id) is None:
            raise TaskNotFoundError(f"Task '{task_id}' does not exist.")

        queue: asyncio.Queue[TaskStatusResponse] = asyncio.Queue(maxsize=256)
        with self._lock:
            self._subscribers.setdefault(task_id, set()).add(queue)
            record = self._records.get(task_id)

        try:
            if record is not None:
                initial = TaskStatusResponse.model_validate(record)
                yield initial
                if initial.status.is_terminal:
                    return

            while True:
                try:
                    snapshot = await asyncio.wait_for(queue.get(), timeout=heartbeat)
                except asyncio.TimeoutError:
                    record = self.find(task_id)
                    if record is None:
                        return
                    snapshot = TaskStatusResponse.model_validate(record)
                    yield snapshot
                    if snapshot.status.is_terminal:
                        return
                    continue

                yield snapshot
                if snapshot.status.is_terminal:
                    return
        finally:
            with self._lock:
                subscribers = self._subscribers.get(task_id)
                if subscribers is not None:
                    subscribers.discard(queue)
                    if not subscribers:
                        self._subscribers.pop(task_id, None)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _evict_locked(self) -> None:
        """Trim the oldest *finished* records beyond ``max_history``."""
        if len(self._records) <= self._max_history:
            return
        for task_id, record in list(self._records.items()):
            if len(self._records) <= self._max_history:
                break
            if record.status.is_terminal:
                self._records.pop(task_id, None)


__all__ = ["TaskContext", "TaskManager", "TaskRecord", "TaskCancelled"]
