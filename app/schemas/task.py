"""Background task status schemas."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class TaskState(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}


class TaskStatusResponse(BaseModel):
    """Snapshot of a background task; also the WebSocket event payload."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    # Validated either from an HTTP payload (``task_id``) or directly from a
    # ``TaskRecord`` dataclass, whose attribute is named ``id``.
    task_id: str = Field(validation_alias=AliasChoices("task_id", "id"))
    name: str
    kind: str = "generic"
    status: TaskState
    progress: float = Field(ge=0.0, le=100.0)
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    started_at: datetime | None = None
    updated_at: datetime
    finished_at: datetime | None = None


__all__ = ["TaskState", "TaskStatusResponse"]
