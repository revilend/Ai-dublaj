"""Shared response envelopes and system schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMModel(BaseModel):
    """Base model able to validate directly from ORM instances."""

    model_config = ConfigDict(from_attributes=True)


class ErrorDetail(BaseModel):
    code: str = Field(examples=["not_found"])
    message: str = Field(examples=["Voice profile 42 does not exist."])
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class TaskAccepted(BaseModel):
    """Returned (HTTP 202) when work has been queued in the background."""

    task_id: str
    status: str
    progress: float = 0.0
    message: str
    status_url: str = Field(description="Poll this URL for progress.")
    websocket_url: str = Field(description="Stream progress events from here.")
    resource_id: int | None = None


class Paginated(BaseModel, Generic[T]):
    items: list[T]
    total: int
    skip: int = 0
    limit: int = 50


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    environment: str
    database: str
    timestamp: datetime


class CapabilityStatus(BaseModel):
    available: bool
    detail: str | None = None


class CapabilitiesResponse(BaseModel):
    """What the current host can actually do — drives the UI badges."""

    device: str
    ffmpeg: CapabilityStatus
    gpu: CapabilityStatus
    tts: CapabilityStatus
    transcription: CapabilityStatus
    translation: CapabilityStatus
    background_separation: CapabilityStatus
    diarization: CapabilityStatus


__all__ = [
    "CapabilitiesResponse",
    "CapabilityStatus",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "ORMModel",
    "Paginated",
    "TaskAccepted",
]
