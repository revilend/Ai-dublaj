"""Instant voice-cloning (text-to-speech) endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, status
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.api.deps import AudioDep, DbSession, StorageDep, TTSDep, TasksDep
from app.core.config import ALLOWED_AUDIO_EXTENSIONS, get_settings
from app.core.exceptions import NotFoundError, ServiceUnavailableError, ValidationError
from app.core.logging import get_logger
from app.models.voice_profile import VoiceProfile
from app.schemas.common import TaskAccepted
from app.schemas.synthesis import QuickTTSRequest, SynthesisResultData
from app.services.storage import StorageManager
from app.services.task_manager import TaskContext
from app.services.tts_engine import SUPPORTED_LANGUAGES, UNSUPPORTED_LANGUAGES

router = APIRouter(prefix="/tts", tags=["synthesis"])
logger = get_logger(__name__)

_REFERENCE_FIELD_NAMES = {"reference_audio", "speaker_wav", "audio", "file", "sample"}


def _clean_form(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop empty form values so optional fields fall back to defaults."""
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        if key != "text" and isinstance(value, str) and not value.strip():
            continue
        cleaned[key] = value
    return cleaned


async def _parse_quick_request(
    request: Request, storage: StorageManager
) -> tuple[QuickTTSRequest, Path | None]:
    """Accept either a JSON body or a multipart upload on the same route."""
    content_type = (request.headers.get("content-type") or "").lower()
    raw: dict[str, Any] = {}
    uploaded_reference: Path | None = None

    if content_type.startswith("application/json"):
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001 - client sent broken JSON
            raise ValidationError("The request body must be valid JSON.") from exc
        if not isinstance(body, dict):
            raise ValidationError("The JSON body must be an object.")
        raw = dict(body)

    elif content_type.startswith(
        ("multipart/form-data", "application/x-www-form-urlencoded")
    ):
        form = await request.form()
        for key, value in form.multi_items():
            if isinstance(value, StarletteUploadFile):
                if key in _REFERENCE_FIELD_NAMES and value.filename:
                    uploaded_reference, _ = await storage.save_upload(
                        value,
                        storage.uploads_dir,
                        allowed_extensions=ALLOWED_AUDIO_EXTENSIONS,
                        prefix="ref-",
                    )
                continue
            raw[key] = value
        raw = _clean_form(raw)

    else:
        raise ValidationError(
            "Unsupported content type. Send application/json or multipart/form-data.",
            details={"content_type": content_type or "unknown"},
        )

    try:
        payload = QuickTTSRequest.model_validate(raw)
    except PydanticValidationError as exc:
        raise ValidationError(
            "Invalid synthesis request.",
            details={"errors": exc.errors(include_url=False)},
        ) from exc

    if uploaded_reference is not None:
        payload.speaker_reference_path = str(uploaded_reference)

    return payload, uploaded_reference


def _resolve_reference(
    payload: QuickTTSRequest, db: Session, storage: StorageManager
) -> Path:
    if payload.speaker_reference_path:
        candidate = storage.absolute(payload.speaker_reference_path)
        if candidate is None or not candidate.exists():
            raise ValidationError("The provided reference audio file does not exist.")
        return candidate

    if payload.voice_profile_id is not None:
        profile = db.get(VoiceProfile, payload.voice_profile_id)
        if profile is None:
            raise NotFoundError(
                f"Voice profile {payload.voice_profile_id} does not exist.",
                details={"voice_profile_id": payload.voice_profile_id},
            )
        if not profile.is_active:
            raise ValidationError(
                f"Voice profile '{profile.name}' is archived and cannot be used."
            )
        candidate = storage.absolute(profile.sample_audio_path)
        if candidate is None or not candidate.exists():
            raise ValidationError(
                f"The reference clip for voice profile '{profile.name}' is missing on disk."
            )
        return candidate

    raise ValidationError(
        "Provide either 'voice_profile_id' or upload a reference audio file.",
        details={"accepted_fields": sorted(_REFERENCE_FIELD_NAMES)},
    )


@router.get("/languages", summary="Language catalogue with availability")
def list_languages() -> dict[str, Any]:
    """Return every language the studio lists, flagged by real availability.

    ``languages`` are the 17 codes XTTS-v2 was trained on. ``unsupported``
    holds languages the UI offers (Uzbek, Kazakh, …) that the current model
    cannot speak — the studio renders them disabled with a reason instead of
    silently producing gibberish.
    """
    settings = get_settings()
    return {
        "default": settings.tts_default_language,
        "model": settings.tts_model_name,
        "languages": [
            {"code": code, "name": name, "supported": True}
            for code, name in sorted(SUPPORTED_LANGUAGES.items())
        ],
        "unsupported": [
            {
                "code": code,
                "name": name,
                "supported": False,
                "reason": "not_trained_by_xtts",
            }
            for code, name in sorted(UNSUPPORTED_LANGUAGES.items())
        ],
    }


@router.post(
    "/quick",
    response_model=TaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Clone a voice and synthesise text",
    description=(
        "Queues a background synthesis job and returns immediately. Poll "
        "`/api/tasks/{task_id}/status` or subscribe to `/ws/tasks/{task_id}`.\n\n"
        "Accepts **application/json** (`voice_profile_id` + `text`) or "
        "**multipart/form-data** (same fields plus a `reference_audio` file)."
    ),
)
async def quick_synthesis(
    request: Request,
    db: DbSession,
    storage: StorageDep,
    audio: AudioDep,
    tts: TTSDep,
    tasks: TasksDep,
) -> TaskAccepted:
    payload, uploaded_reference = await _parse_quick_request(request, storage)
    try:
        reference = _resolve_reference(payload, db, storage)
    except Exception:
        if uploaded_reference is not None:
            storage.delete(uploaded_reference)
        raise

    output_suffix = ".mp3" if payload.output_format == "mp3" else ".wav"
    record = tasks.create(
        f"Quick TTS · {payload.text[:48]}",
        kind="tts",
        metadata={
            "voice_profile_id": payload.voice_profile_id,
            "language": payload.language,
            "characters": len(payload.text),
            "speed": payload.speed,
            "pitch": payload.pitch,
        },
    )
    output_path = storage.outputs_dir / f"tts-{record.id}{output_suffix}"

    async def body(ctx: TaskContext) -> dict[str, Any]:
        ctx.report(2.0, "Warming up the voice model…")
        result = await tts.clone_voice_async(
            text=payload.text,
            reference_audio=reference,
            language=payload.language,
            speed=payload.speed,
            output_path=output_path,
            normalize=payload.normalize,
            pitch=payload.pitch,
            storage=storage,
            progress_callback=lambda value, message: ctx.report(
                min(4.0 + value * 0.9, 94.0), message
            ),
        )

        produced = result.audio_path
        if payload.output_format == "mp3":
            if not audio.ffmpeg_available:
                raise ServiceUnavailableError(
                    "MP3 output requires ffmpeg, which is not installed on this host.",
                    details={"requested_format": "mp3"},
                )
            ctx.report(97.0, "Encoding MP3…")
            mp3_path = produced.with_suffix(".mp3")
            await asyncio.to_thread(audio.convert_audio, produced, mp3_path)
            produced = mp3_path

        data = SynthesisResultData(
            audio_path=storage.relative(produced),
            audio_url=storage.url_for(produced) or "",
            duration_seconds=result.duration_seconds,
            language=result.language,
            voice_profile_id=payload.voice_profile_id,
            chunks=result.chunks,
            speed=payload.speed,
            pitch=payload.pitch,
            output_format=payload.output_format,
        )
        payload_dict = data.model_dump()
        ctx.complete(payload_dict, message="Synthesis complete.")
        return payload_dict

    tasks.start(record, body)

    return TaskAccepted(
        task_id=record.id,
        status=record.status.value,
        progress=record.progress,
        message="Synthesis queued.",
        status_url=f"/api/tasks/{record.id}/status",
        websocket_url=f"/ws/tasks/{record.id}",
    )


__all__ = ["router"]
