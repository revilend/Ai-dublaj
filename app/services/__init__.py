"""Service layer: engines, storage and background task orchestration.

All heavy objects (the XTTS model, Whisper, NLLB, the task registry) are
process-wide singletons built lazily through the ``get_*`` factories below, so
routers can depend on them without paying for model loads at import time.
"""

from __future__ import annotations

from functools import lru_cache

from app.services.audio_processor import AudioProcessor, MediaInfo, SeparatedStems
from app.services.diarization_service import (
    DiarizationService,
    DiarizationUnavailable,
    SpeakerTurn,
)
from app.services.dubbing_service import STEP_MESSAGES, STEP_TOTAL, DubbingService
from app.services.storage import StorageManager
from app.services.subtitle_parser import SubtitleCue, parse_subtitles, to_srt
from app.services.task_manager import TaskContext, TaskManager, TaskRecord
from app.services.transcription_service import (
    TranscriptSegment,
    TranscriptionService,
    TranscriptionUnavailable,
)
from app.services.translation_service import TranslationService, TranslationUnavailable
from app.services.tts_engine import (
    SynthesisResult,
    TTSEngine,
    TTSEngineUnavailable,
    TTSValidationError,
)


@lru_cache(maxsize=1)
def get_storage() -> StorageManager:
    return StorageManager()


@lru_cache(maxsize=1)
def get_audio_processor() -> AudioProcessor:
    return AudioProcessor()


@lru_cache(maxsize=1)
def get_tts_engine() -> TTSEngine:
    return TTSEngine(audio_processor=get_audio_processor())


@lru_cache(maxsize=1)
def get_transcription_service() -> TranscriptionService:
    return TranscriptionService()


@lru_cache(maxsize=1)
def get_translation_service() -> TranslationService:
    return TranslationService()


@lru_cache(maxsize=1)
def get_diarization_service() -> DiarizationService:
    return DiarizationService()


@lru_cache(maxsize=1)
def get_task_manager() -> TaskManager:
    """In-memory task registry. Bound to the event loop during startup."""
    return TaskManager()


@lru_cache(maxsize=1)
def get_dubbing_service() -> DubbingService:
    return DubbingService(
        storage=get_storage(),
        audio=get_audio_processor(),
        tts=get_tts_engine(),
        transcriber=get_transcription_service(),
        translator=get_translation_service(),
        diarizer=get_diarization_service(),
    )


__all__ = [
    "STEP_MESSAGES",
    "STEP_TOTAL",
    "AudioProcessor",
    "DiarizationService",
    "DiarizationUnavailable",
    "DubbingService",
    "MediaInfo",
    "SeparatedStems",
    "SpeakerTurn",
    "StorageManager",
    "SubtitleCue",
    "SynthesisResult",
    "TTSEngine",
    "TTSEngineUnavailable",
    "TTSValidationError",
    "TaskContext",
    "TaskManager",
    "TaskRecord",
    "TranscriptSegment",
    "TranscriptionService",
    "TranscriptionUnavailable",
    "TranslationService",
    "TranslationUnavailable",
    "get_audio_processor",
    "get_diarization_service",
    "get_dubbing_service",
    "get_storage",
    "get_task_manager",
    "get_transcription_service",
    "get_translation_service",
    "get_tts_engine",
    "parse_subtitles",
    "to_srt",
]
