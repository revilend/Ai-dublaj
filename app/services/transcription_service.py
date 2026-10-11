"""Optional speech-to-text service (faster-whisper).

Used by the dubbing pipeline to turn the extracted audio into timed subtitle
segments. The model is loaded lazily and cached; when the library is not
installed the service reports itself as unavailable instead of crashing.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import ServiceUnavailableError, ValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)


class TranscriptionUnavailable(ServiceUnavailableError):
    code = "transcription_unavailable"
    default_message = "Speech recognition is not available on this host."


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


class TranscriptionService:
    """Thin wrapper over ``faster_whisper.WhisperModel``."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def dependency_status(self) -> tuple[bool, str | None]:
        try:
            import faster_whisper  # noqa: F401
        except Exception:  # noqa: BLE001
            return False, (
                "faster-whisper is not installed. Install the local AI runtime "
                "with `pip install -r requirements-ai.txt`."
            )
        return True, None

    def detail(self) -> str:
        available, reason = self.dependency_status()
        if available:
            return f"faster-whisper '{self.settings.whisper_model_size}' ready"
        return reason or "unavailable"

    def _load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model

            available, reason = self.dependency_status()
            if not available:
                raise TranscriptionUnavailable(reason or "Whisper is unavailable.")

            from faster_whisper import WhisperModel

            device = "cpu"
            compute_type = "int8"
            try:
                import torch

                if torch.cuda.is_available():  # pragma: no cover - GPU hosts only
                    device, compute_type = "cuda", "float16"
            except Exception:  # noqa: BLE001
                pass

            logger.info(
                "Loading Whisper '%s' on %s (%s)…",
                self.settings.whisper_model_size, device, compute_type,
            )
            self._model = WhisperModel(
                self.settings.whisper_model_size, device=device, compute_type=compute_type
            )
            return self._model

    # ------------------------------------------------------------------ #
    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[TranscriptSegment]:
        """Return timed transcript segments for a local audio file."""
        source = Path(audio_path)
        if not source.exists():
            raise ValidationError(f"Audio file '{source.name}' does not exist.")

        model = self._load()
        target_language = None if (language or "auto") in ("auto", "") else language

        if progress_callback:
            progress_callback(30.0, "Transcribing audio…")

        raw_segments, info = model.transcribe(
            str(source),
            language=target_language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        results: list[TranscriptSegment] = []
        for item in raw_segments:
            text = (item.text or "").strip()
            if not text:
                continue
            results.append(
                TranscriptSegment(
                    start=float(item.start or 0.0),
                    end=float(item.end or 0.0),
                    text=text,
                )
            )

        logger.info(
            "Transcribed %s -> %d segment(s) (detected language: %s)",
            source.name, len(results), getattr(info, "language", target_language or "unknown"),
        )
        return results

    async def transcribe_async(self, *args: Any, **kwargs: Any) -> list[TranscriptSegment]:
        return await asyncio.to_thread(self.transcribe, *args, **kwargs)


__all__ = ["TranscriptionService", "TranscriptionUnavailable", "TranscriptSegment"]
