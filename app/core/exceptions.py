"""Domain-level exceptions translated into HTTP responses by app/main.py."""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for expected, user-facing application errors.

    Handlers in ``app.main`` convert these into a consistent JSON envelope::

        {"error": {"code": "...", "message": "...", "details": {...}}}
    """

    status_code: int = 500
    code: str = "internal_error"
    default_message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.details: dict[str, Any] = details or {}
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class ValidationError(AppError):
    """The request was syntactically valid but semantically rejected."""

    status_code = 422
    code = "validation_error"
    default_message = "The provided input is not valid."


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
    default_message = "The requested resource was not found."


class ConflictError(AppError):
    status_code = 409
    code = "conflict"
    default_message = "The resource already exists."


class StorageError(AppError):
    """Raised when persisting an upload or artifact fails."""

    status_code = 500
    code = "storage_error"
    default_message = "Failed to store the uploaded file."


class MediaProcessingError(AppError):
    """Raised when ffmpeg/ffprobe cannot process a media file."""

    status_code = 422
    code = "media_processing_error"
    default_message = "The media file could not be processed."


class ServiceUnavailableError(AppError):
    """Raised when an optional engine (TTS, Whisper, ...) is not installed."""

    status_code = 503
    code = "service_unavailable"
    default_message = "A required service is not available on this host."


class TaskNotFoundError(NotFoundError):
    code = "task_not_found"
    default_message = "The requested task does not exist."


__all__ = [
    "AppError",
    "ConflictError",
    "MediaProcessingError",
    "NotFoundError",
    "ServiceUnavailableError",
    "StorageError",
    "TaskNotFoundError",
    "ValidationError",
]
