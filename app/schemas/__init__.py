"""Pydantic request/response schemas (API contract)."""

from app.schemas.common import (
    CapabilitiesResponse,
    CapabilityStatus,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ORMModel,
    Paginated,
    TaskAccepted,
)
from app.schemas.dubbing import (
    DubbingProjectCreate,
    DubbingProjectDetail,
    DubbingProjectRead,
    RenderRequest,
    SegmentBulkCreate,
    SubtitleSegmentCreate,
    SubtitleSegmentRead,
    SubtitleSegmentUpdate,
)
from app.schemas.synthesis import QuickTTSRequest, SynthesisResultData
from app.schemas.task import TaskState, TaskStatusResponse
from app.schemas.voice import (
    VoiceProfileCreate,
    VoiceProfileRead,
    VoiceProfileUpdate,
)

__all__ = [
    "CapabilitiesResponse",
    "CapabilityStatus",
    "DubbingProjectCreate",
    "DubbingProjectDetail",
    "DubbingProjectRead",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "ORMModel",
    "Paginated",
    "QuickTTSRequest",
    "RenderRequest",
    "SegmentBulkCreate",
    "SubtitleSegmentCreate",
    "SubtitleSegmentRead",
    "SubtitleSegmentUpdate",
    "SynthesisResultData",
    "TaskAccepted",
    "TaskState",
    "TaskStatusResponse",
    "VoiceProfileCreate",
    "VoiceProfileRead",
    "VoiceProfileUpdate",
]
