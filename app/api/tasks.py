"""Background task endpoints: polling, listing, cancellation and WebSocket."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from app.api.deps import PaginationDep, TasksDep
from app.core.exceptions import TaskNotFoundError
from app.core.logging import get_logger
from app.schemas.common import Paginated
from app.schemas.task import TaskState, TaskStatusResponse

router = APIRouter(prefix="/tasks", tags=["tasks"])
# Mounted at the application root so clients can use the friendlier
# `/ws/tasks/{id}` path instead of `/api/tasks/{id}/stream`.
ws_router = APIRouter(tags=["tasks"])
logger = get_logger(__name__)

WS_TASK_NOT_FOUND = 4404


@router.get("", response_model=Paginated[TaskStatusResponse], summary="List background tasks")
def list_tasks(
    tasks: TasksDep,
    pagination: PaginationDep,
    task_status: Annotated[TaskState | None, Query(alias="status")] = None,
    kind: Annotated[str | None, Query(description="Filter by task kind, e.g. 'tts'.")] = None,
) -> Paginated[TaskStatusResponse]:
    records = tasks.list(limit=10_000, status=task_status, kind=kind)
    total = len(records)
    page = records[pagination.skip : pagination.skip + pagination.limit]
    return Paginated[TaskStatusResponse](
        items=[TaskStatusResponse.model_validate(record) for record in page],
        total=total,
        skip=pagination.skip,
        limit=pagination.limit,
    )


@router.get("/stats", summary="Task registry statistics")
def task_stats(tasks: TasksDep) -> dict[str, int]:
    return tasks.stats()


@router.get(
    "/{task_id}/status",
    response_model=TaskStatusResponse,
    summary="Real-time task status",
    description="Poll this endpoint for progress (0–100 %).",
)
def task_status(task_id: str, tasks: TasksDep) -> TaskStatusResponse:
    record = tasks.get(task_id)
    return TaskStatusResponse.model_validate(record)


@router.delete(
    "/{task_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel a running task",
)
def cancel_task(task_id: str, tasks: TasksDep) -> dict[str, object]:
    record = tasks.get(task_id)
    cancelled = tasks.cancel(task_id)
    return {
        "task_id": task_id,
        "cancelled": cancelled,
        "status": record.status.value,
        "message": (
            "Cancellation requested."
            if cancelled
            else "The task had already finished."
        ),
    }


@ws_router.websocket("/ws/tasks/{task_id}")
async def stream_task(websocket: WebSocket, task_id: str) -> None:
    """Push status snapshots as they happen (replaces polling)."""
    await websocket.accept()
    tasks = websocket.app.state.task_manager

    try:
        async for snapshot in tasks.subscribe(task_id):
            await websocket.send_json(snapshot.model_dump(mode="json"))
            if snapshot.status.is_terminal:
                break
    except TaskNotFoundError:
        await websocket.send_json(
            {"error": {"code": "task_not_found", "message": f"Task '{task_id}' does not exist."}}
        )
        await websocket.close(code=WS_TASK_NOT_FOUND)
        return
    except WebSocketDisconnect:
        logger.debug("Client disconnected from task stream %s", task_id)
        return
    except asyncio.CancelledError:  # pragma: no cover - server shutdown
        raise
    except Exception:  # noqa: BLE001 - never leak tracebacks over a socket
        logger.exception("Task stream %s failed", task_id)
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


__all__ = ["router", "ws_router"]
