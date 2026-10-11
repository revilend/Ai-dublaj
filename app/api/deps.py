"""Reusable FastAPI dependencies (session, engines, pagination)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Query
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.services import (
    AudioProcessor,
    DiarizationService,
    DubbingService,
    StorageManager,
    TaskManager,
    TranscriptionService,
    TranslationService,
    TTSEngine,
    get_audio_processor,
    get_diarization_service,
    get_dubbing_service,
    get_storage,
    get_task_manager,
    get_transcription_service,
    get_translation_service,
    get_tts_engine,
)

DbSession = Annotated[Session, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
StorageDep = Annotated[StorageManager, Depends(get_storage)]
AudioDep = Annotated[AudioProcessor, Depends(get_audio_processor)]
TTSDep = Annotated[TTSEngine, Depends(get_tts_engine)]
TranscriberDep = Annotated[TranscriptionService, Depends(get_transcription_service)]
TranslatorDep = Annotated[TranslationService, Depends(get_translation_service)]
DiarizerDep = Annotated[DiarizationService, Depends(get_diarization_service)]
TasksDep = Annotated[TaskManager, Depends(get_task_manager)]
DubbingDep = Annotated[DubbingService, Depends(get_dubbing_service)]


@dataclass(slots=True)
class Pagination:
    """Standard offset/limit pagination query parameters."""

    skip: int = Query(default=0, ge=0, description="Number of records to skip.")
    limit: int = Query(default=50, ge=1, le=200, description="Page size.")


PaginationDep = Annotated[Pagination, Depends()]

__all__ = [
    "AudioDep",
    "DbSession",
    "DiarizerDep",
    "DubbingDep",
    "Pagination",
    "PaginationDep",
    "SettingsDep",
    "StorageDep",
    "TTSDep",
    "TasksDep",
    "TranscriberDep",
    "TranslatorDep",
]
