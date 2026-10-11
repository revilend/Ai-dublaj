"""Dubbing project / subtitle segment schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from app.models.dubbing_project import DubbingStatus
from app.schemas.common import ORMModel


# --------------------------------------------------------------------------- #
# Subtitle segments
# --------------------------------------------------------------------------- #
class SubtitleSegmentBase(ORMModel):
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    original_text: str = Field(default="", max_length=10000)
    translated_text: str | None = Field(default=None, max_length=10000)
    speaker_id: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _validate_span(self) -> "SubtitleSegmentBase":
        if self.end_time < self.start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        return self


class SubtitleSegmentCreate(SubtitleSegmentBase):
    pass


class SubtitleSegmentUpdate(ORMModel):
    """Partial update. Omitted fields are untouched; sending ``null`` clears."""

    start_time: float | None = Field(default=None, ge=0.0)
    end_time: float | None = Field(default=None, ge=0.0)
    order_index: int | None = None
    original_text: str | None = Field(default=None, max_length=10000)
    translated_text: str | None = Field(default=None, max_length=10000)
    speaker_id: str | None = Field(default=None, max_length=64)
    voice_profile_id: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Assign a saved voice profile to this segment only. Send null to fall "
            "back to the project's voice."
        ),
    )


class SubtitleSegmentRead(SubtitleSegmentBase):
    id: int
    project_id: int
    order_index: int = 0
    dubbed_audio_path: str | None = None
    dubbed_audio_url: str | None = None
    voice_profile_id: int | None = None
    voice_profile_name: str | None = None
    created_at: datetime


class SegmentBulkCreate(BaseModel):
    segments: list[SubtitleSegmentCreate] = Field(min_length=1, max_length=5000)


# --------------------------------------------------------------------------- #
# Dubbing projects
# --------------------------------------------------------------------------- #
class DubbingProjectCreate(ORMModel):
    title: str = Field(min_length=1, max_length=200)
    source_language: str = Field(default="auto", min_length=2, max_length=16)
    target_language: str = Field(default="en", min_length=2, max_length=16)
    voice_profile_id: int | None = Field(default=None, gt=0)


class DubbingProjectRead(ORMModel):
    id: int
    title: str
    status: DubbingStatus
    progress: float = 0.0
    error_message: str | None = None
    duration_seconds: float | None = None
    source_language: str
    target_language: str
    voice_profile_id: int | None = None
    video_path: str | None = None
    original_audio_path: str | None = None
    vocals_audio_path: str | None = None
    background_audio_path: str | None = None
    dubbed_audio_path: str | None = None
    final_video_path: str | None = None
    segment_count: int = 0
    last_task_id: str | None = None
    video_url: str | None = None
    final_video_url: str | None = None
    vocals_audio_url: str | None = None
    background_audio_url: str | None = None
    dubbed_audio_url: str | None = None
    created_at: datetime
    updated_at: datetime


class DubbingProjectDetail(DubbingProjectRead):
    segments: list[SubtitleSegmentRead] = Field(default_factory=list)


class DubbingUploadResponse(BaseModel):
    """Returned by ``POST /api/dubbing/upload``.

    The pipeline runs in the background, so the project id comes back
    immediately and the parsed segments become available once the analysis task
    completes (follow ``status_url`` or ``websocket_url``, then
    ``GET /api/dubbing/{project_id}``).
    """

    project_id: int
    title: str
    status: DubbingStatus
    task_id: str | None = Field(
        default=None, description="Null when the upload was created without analysis."
    )
    status_url: str | None = None
    websocket_url: str | None = None
    message: str


class SegmentPreviewResponse(BaseModel):
    """Result of ``POST /api/dubbing/segments/{id}/preview``."""

    segment_id: int
    project_id: int
    dubbed_audio_path: str
    dubbed_audio_url: str | None = None
    duration_seconds: float | None = None


class TranslateResponse(BaseModel):
    """Result of ``POST /api/dubbing/{project_id}/translate``."""

    project_id: int
    translated: int
    segments: int


class RenderRequest(BaseModel):
    """Optional overrides for ``POST /api/dubbing/{id}/render``."""

    voice_profile_id: int | None = Field(default=None, gt=0)
    speaker_reference_path: str | None = None
    target_language: str | None = Field(default=None, min_length=2, max_length=16)
    translate: bool = Field(
        default=True, description="Translate segments before synthesis when languages differ."
    )
    mix_background: bool = Field(
        default=True, description="Mix separated background audio under the dub."
    )


__all__ = [
    "DubbingProjectCreate",
    "DubbingProjectDetail",
    "DubbingProjectRead",
    "DubbingUploadResponse",
    "RenderRequest",
    "SegmentBulkCreate",
    "SegmentPreviewResponse",
    "SubtitleSegmentCreate",
    "SubtitleSegmentRead",
    "SubtitleSegmentUpdate",
    "TranslateResponse",
]
