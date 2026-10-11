"""Health and capability endpoints.

``/api/system/capabilities`` lets the studio UI show exactly which local engines
are installed, so users understand why a button may be disabled.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import (
    AudioDep,
    DbSession,
    DiarizerDep,
    TasksDep,
    TTSDep,
    TranscriberDep,
    TranslatorDep,
)
from app.core.config import get_settings
from app.schemas.common import (
    CapabilitiesResponse,
    CapabilityStatus,
    HealthResponse,
)

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health(db: DbSession) -> HealthResponse:
    settings = get_settings()
    database_state = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - report rather than crash
        database_state = "unavailable"

    return HealthResponse(
        status="ok" if database_state == "ok" else "degraded",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        database=database_state,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/system/capabilities",
    response_model=CapabilitiesResponse,
    summary="Which local engines are available",
)
def capabilities(
    audio: AudioDep,
    tts: TTSDep,
    transcriber: TranscriberDep,
    translator: TranslatorDep,
    diarizer: DiarizerDep,
    tasks: TasksDep,
) -> CapabilitiesResponse:
    tts_available, tts_reason = tts.dependency_status()
    asr_available, asr_reason = transcriber.dependency_status()
    mt_available, mt_reason = translator.dependency_status()
    diar_available, diar_reason = diarizer.dependency_status()

    device = tts.resolve_device()
    gpu_ready = device == "cuda"

    return CapabilitiesResponse(
        device=device,
        ffmpeg=CapabilityStatus(
            available=audio.ffmpeg_available and audio.ffprobe_available,
            detail=audio.capability_detail(),
        ),
        gpu=CapabilityStatus(
            available=gpu_ready,
            detail="CUDA device detected" if gpu_ready else "Running on CPU",
        ),
        tts=CapabilityStatus(
            available=tts_available,
            detail=tts.dependency_detail() if tts_available else tts_reason,
        ),
        transcription=CapabilityStatus(
            available=asr_available,
            detail=transcriber.detail() if asr_available else asr_reason,
        ),
        translation=CapabilityStatus(
            available=mt_available,
            detail=translator.detail() if mt_available else mt_reason,
        ),
        background_separation=CapabilityStatus(
            available=audio.background_separation_available,
            detail=(
                "Demucs ready"
                if audio.background_separation_available
                else "Demucs not installed — the original audio bed is kept"
            ),
        ),
        diarization=CapabilityStatus(
            available=diar_available,
            detail=diarizer.detail() if diar_available else diar_reason,
        ),
    )


__all__ = ["router"]
