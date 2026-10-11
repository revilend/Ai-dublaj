"""Text-to-speech request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class QuickTTSRequest(BaseModel):
    """JSON body for ``POST /api/tts/quick``.

    The same endpoint also accepts a ``multipart/form-data`` upload when the
    caller wants to clone from a freshly uploaded reference clip instead of a
    saved :class:`VoiceProfile`.
    """

    text: str = Field(min_length=1, max_length=5000)
    language: str = Field(default="en", min_length=2, max_length=16)
    speed: float = Field(default=1.0, ge=0.25, le=3.0)
    pitch: float = Field(
        default=0.0,
        ge=-12.0,
        le=12.0,
        description="Pitch shift in semitones, applied after synthesis (duration preserved).",
    )
    voice_profile_id: int | None = Field(default=None, gt=0)
    speaker_reference_path: str | None = Field(
        default=None,
        description="Server-side path to a reference clip (used by the API when "
        "an audio file was uploaded with the request).",
    )
    output_format: Literal["wav", "mp3"] = "wav"
    normalize: bool = Field(default=False, description="Apply EBU R128 loudness normalisation.")

    @field_validator("text")
    @classmethod
    def _require_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be blank")
        return stripped

    @field_validator("language")
    @classmethod
    def _normalise_language(cls, value: str) -> str:
        return value.strip().lower()


class SynthesisResultData(BaseModel):
    audio_path: str
    audio_url: str
    duration_seconds: float | None = None
    language: str
    voice_profile_id: int | None = None
    chunks: int = 1
    speed: float = 1.0
    pitch: float = 0.0
    output_format: str = "wav"


__all__ = ["QuickTTSRequest", "SynthesisResultData"]
