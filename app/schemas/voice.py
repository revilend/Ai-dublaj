"""Voice profile request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import Field, field_validator

from app.schemas.common import ORMModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.voice_profile import VoiceProfile


class VoiceProfileBase(ORMModel):
    name: str = Field(min_length=1, max_length=150, examples=["Narrator — warm male"])
    description: str | None = Field(default=None, max_length=2000)
    language: str = Field(default="en", min_length=2, max_length=16)

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("name must not be empty")
        return cleaned

    @field_validator("language")
    @classmethod
    def _normalise_language(cls, value: str) -> str:
        return value.strip().lower()


class VoiceProfileCreate(VoiceProfileBase):
    """JSON body variant (the HTTP endpoint also accepts multipart uploads)."""


class VoiceProfileUpdate(ORMModel):
    name: str | None = Field(default=None, min_length=1, max_length=150)
    description: str | None = Field(default=None, max_length=2000)
    language: str | None = Field(default=None, min_length=2, max_length=16)
    is_active: bool | None = None


class VoiceProfileRead(VoiceProfileBase):
    id: int
    sample_audio_path: str
    sample_audio_url: str | None = None
    duration_seconds: float | None = None
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(
        cls, voice: "VoiceProfile", sample_audio_url: str | None = None
    ) -> "VoiceProfileRead":
        payload = cls.model_validate(voice)
        payload.sample_audio_url = sample_audio_url
        return payload


__all__ = [
    "VoiceProfileBase",
    "VoiceProfileCreate",
    "VoiceProfileRead",
    "VoiceProfileUpdate",
]
