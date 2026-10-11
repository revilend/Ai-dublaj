"""Optional speaker diarization ("who spoke when").

Diarization needs a dedicated model — Whisper's transcript does not carry
speaker labels. The reference implementation is ``pyannote.audio``, whose
pretrained pipelines live in a **gated** Hugging Face repository: the user has
to accept the model terms and supply a token.

That is a deliberate design choice here: when pyannote is not available we
report ourselves *unavailable* and leave ``speaker_id`` as ``None``. Guessing
speakers from a pitch heuristic would silently label segments wrong, which is
worse than admitting the gap — the same contract the TTS/Whisper/NLLB/Demucs
engines follow.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import ServiceUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


class DiarizationUnavailable(ServiceUnavailableError):
    code = "diarization_unavailable"
    default_message = "Speaker diarization is not available on this host."


@dataclass(slots=True)
class SpeakerTurn:
    """One contiguous span attributed to a speaker."""

    start: float
    end: float
    speaker: str


class DiarizationService:
    """Assigns speaker labels to already-timestamped transcript segments."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._pipeline: Any | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #
    def dependency_status(self) -> tuple[bool, str | None]:
        if not self.settings.enable_diarization:
            return False, "Diarization is disabled (VOX_ENABLE_DIARIZATION=false)."
        try:
            import pyannote.audio  # noqa: F401
        except Exception:  # noqa: BLE001
            return False, (
                "pyannote.audio is not installed. Diarization is optional: install it "
                "(`pip install pyannote.audio`) and set VOX_HF_TOKEN after accepting the "
                "model terms on Hugging Face."
            )
        if not self.settings.has_hf_token:
            return False, (
                "pyannote.audio is installed but VOX_HF_TOKEN is not set; the "
                f"'{self.settings.diarization_model}' pipeline is a gated model."
            )
        return True, None

    def detail(self) -> str:
        available, reason = self.dependency_status()
        if available:
            return f"{self.settings.diarization_model} ready"
        return reason or "unavailable"

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _load(self) -> Any:
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            available, reason = self.dependency_status()
            if not available:
                raise DiarizationUnavailable(reason or "Diarization is unavailable.")

            from pyannote.audio import Pipeline

            logger.info("Loading diarization pipeline %s…", self.settings.diarization_model)
            token = self.settings.hf_token
            try:
                self._pipeline = Pipeline.from_pretrained(
                    self.settings.diarization_model, token=token
                )
            except TypeError:  # pragma: no cover - older pyannote signatures
                self._pipeline = Pipeline.from_pretrained(
                    self.settings.diarization_model, use_auth_token=token
                )
            return self._pipeline

    def timeline(self, audio_path: str | Path) -> list[SpeakerTurn]:
        """Run diarization and return the speaker turns found in ``audio_path``."""
        source = Path(audio_path)
        if not source.exists():
            raise DiarizationUnavailable(f"Audio file '{source.name}' does not exist.")

        pipeline = self._load()
        annotation = pipeline(str(source))

        turns: list[SpeakerTurn] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
            )
        turns.sort(key=lambda item: item.start)
        logger.info("Diarization found %d turn(s) in %s", len(turns), source.name)
        return turns

    def assign_speakers(
        self, audio_path: str | Path, spans: list[tuple[float, float]]
    ) -> list[str | None]:
        """Label each ``(start, end)`` span with the overlapping speaker.

        Returns ``[None] * len(spans)`` (without raising) when diarization is
        unavailable, so the dubbing pipeline can continue single-voiced.
        """
        if not spans:
            return []

        try:
            turns = self.timeline(audio_path)
        except DiarizationUnavailable as exc:
            logger.info("Skipping diarization: %s", exc.message)
            return [None] * len(spans)
        except Exception:  # noqa: BLE001 - a diarization failure must not kill a dub
            logger.exception("Diarization failed; continuing without speaker labels")
            return [None] * len(spans)

        if not turns:
            return [None] * len(spans)

        labels: list[str | None] = []
        for start, end in spans:
            best_speaker: str | None = None
            best_overlap = 0.0
            for turn in turns:
                overlap = min(end, turn.end) - max(start, turn.start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn.speaker
            # Require a real overlap rather than snapping to the nearest turn.
            labels.append(best_speaker if best_overlap > 0 else None)
        return labels

    async def assign_speakers_async(
        self, audio_path: str | Path, spans: list[tuple[float, float]]
    ) -> list[str | None]:
        return await asyncio.to_thread(self.assign_speakers, audio_path, spans)


__all__ = ["DiarizationService", "DiarizationUnavailable", "SpeakerTurn"]
