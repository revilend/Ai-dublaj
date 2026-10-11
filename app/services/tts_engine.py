"""Coqui XTTS-v2 wrapper with CUDA acceleration and CPU fallback.

The engine is deliberately *lazy*: importing this module never imports PyTorch
or Coqui TTS, so the API boots in milliseconds on a CPU-only host. The model is
loaded on first synthesis and cached for the lifetime of the process.

Long inputs are split into sentence-sized chunks because XTTS degrades on very
long prompts; the chunks are synthesised individually and concatenated.
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.config import Settings, get_settings
from app.core.exceptions import ServiceUnavailableError, ValidationError
from app.core.logging import get_logger
from app.services.audio_processor import AudioProcessor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.services.storage import StorageManager

logger = get_logger(__name__)

DEFAULT_MAX_CHUNK_CHARS = 250

# XTTS-v2 supported target languages.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "cs": "Czech",
    "ar": "Arabic",
    "zh-cn": "Chinese (Mandarin)",
    "ja": "Japanese",
    "hu": "Hungarian",
    "ko": "Korean",
    "hi": "Hindi",
}

# Languages we display in the studio but that XTTS-v2 was never trained on.
# Asking the model for them would silently produce gibberish, so we reject them
# with an explicit, actionable error instead.
UNSUPPORTED_LANGUAGES: dict[str, str] = {
    "uz": "Uzbek",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "tg": "Tajik",
    "az": "Azerbaijani",
}

_LANGUAGE_ALIASES = {
    "zh": "zh-cn",
    "zh-hans": "zh-cn",
    "zh_cn": "zh-cn",
    "cmn": "zh-cn",
    "en-us": "en",
    "en-gb": "en",
    "pt-br": "pt",
    "pt-pt": "pt",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。！？؛])\s+")


class TTSEngineUnavailable(ServiceUnavailableError):
    code = "tts_unavailable"
    default_message = "The local text-to-speech engine is not available on this host."


class TTSValidationError(ValidationError):
    code = "tts_invalid_input"


@dataclass(slots=True)
class SynthesisResult:
    """Outcome of one ``clone_voice`` call."""

    audio_path: Path
    duration_seconds: float | None
    language: str
    chunks: int
    device: str
    sample_rate: int


def normalize_language(language: str | None, *, default: str = "en") -> str:
    """Map user input onto an XTTS-supported language code."""
    candidate = (language or default).strip().lower().replace(" ", "")
    candidate = _LANGUAGE_ALIASES.get(candidate, candidate)
    if candidate in UNSUPPORTED_LANGUAGES:
        raise TTSValidationError(
            f"XTTS-v2 has no voice for {UNSUPPORTED_LANGUAGES[candidate]}. "
            "It supports 17 languages only — translate the text first or use a "
            "language-specific checkpoint.",
            details={
                "requested": candidate,
                "reason": "not_trained_by_xtts",
                "supported": sorted(SUPPORTED_LANGUAGES),
            },
        )
    if candidate not in SUPPORTED_LANGUAGES:
        raise TTSValidationError(
            f"Language '{language}' is not supported by XTTS-v2.",
            details={
                "requested": language,
                "supported": sorted(SUPPORTED_LANGUAGES),
            },
        )
    return candidate


class TTSEngine:
    """Voice cloning façade around ``TTS.api.TTS`` (XTTS-v2)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        audio_processor: AudioProcessor | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.audio = audio_processor or AudioProcessor(self.settings)
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Availability / device
    # ------------------------------------------------------------------ #
    def dependency_status(self) -> tuple[bool, str | None]:
        """``(available, reason)`` — never raises, never imports torch twice."""
        try:
            import TTS  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, (
                "Coqui TTS is not installed. Install the local AI runtime with "
                "`pip install -r requirements-ai.txt`."
            )
        except BaseException:  # pragma: no cover - defensive
            return False, "Coqui TTS failed to import."
        return True, None

    def dependency_detail(self) -> str:
        available, reason = self.dependency_status()
        if available:
            return f"XTTS-v2 ready on {self.resolve_device()}"
        return reason or "unavailable"

    def resolve_device(self) -> str:
        """Honour the configured device, falling back to CPU when needed."""
        preference = self.settings.tts_device
        if preference == "cpu":
            return "cpu"

        try:
            import torch
        except Exception:  # noqa: BLE001 - torch missing == CPU only
            if preference == "cuda":
                logger.warning("CUDA was requested but PyTorch is not installed; using CPU.")
            return "cpu"

        cuda_ready = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        if cuda_ready:
            return "cuda"
        if preference == "cuda":
            logger.warning("CUDA was requested but no GPU is available; falling back to CPU.")
        return "cpu"

    @property
    def device(self) -> str:
        return self.resolve_device()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #
    def load_model(self) -> Any:
        """Load XTTS-v2 once, thread-safely."""
        with self._model_lock:
            if self._model is not None:
                return self._model

            available, reason = self.dependency_status()
            if not available:
                raise TTSEngineUnavailable(reason or "Coqui TTS is unavailable.")

            from TTS.api import TTS  # imported lazily: heavy dependency

            device = self.resolve_device()
            logger.info("Loading XTTS model '%s' on %s…", self.settings.tts_model_name, device)
            model = TTS(self.settings.tts_model_name)
            try:
                model.to(device)
            except Exception:  # noqa: BLE001 - some versions expose no .to()
                logger.debug("Model has no .to(); relying on torch device selection.")
            self._model = model
            logger.info("XTTS model ready on %s.", device)
            return self._model

    def unload(self) -> None:
        """Free the model (useful before a large video render)."""
        with self._model_lock:
            self._model = None
        try:
            import torch

            if torch.cuda.is_available():  # pragma: no cover - GPU hosts only
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        logger.info("XTTS model unloaded.")

    # ------------------------------------------------------------------ #
    # Text chunking
    # ------------------------------------------------------------------ #
    @staticmethod
    def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
        """Split text on sentence boundaries, hard-wrapping long sentences."""
        cleaned = " ".join(text.split())
        if not cleaned:
            return []
        if len(cleaned) <= max_chars:
            return [cleaned]

        chunks: list[str] = []
        buffer = ""
        for sentence in _SENTENCE_SPLIT.split(cleaned):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_chars:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                words = sentence.split(" ")
                piece = ""
                for word in words:
                    if piece and len(piece) + len(word) + 1 > max_chars:
                        chunks.append(piece)
                        piece = word
                    else:
                        piece = f"{piece} {word}".strip()
                if piece:
                    buffer = piece
                continue
            if buffer and len(buffer) + len(sentence) + 1 > max_chars:
                chunks.append(buffer)
                buffer = sentence
            else:
                buffer = f"{buffer} {sentence}".strip()
        if buffer:
            chunks.append(buffer)
        return [chunk for chunk in chunks if chunk]

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #
    def clone_voice(
        self,
        text: str,
        reference_audio: str | Path,
        language: str = "en",
        speed: float = 1.0,
        output_path: str | Path | None = None,
        *,
        normalize: bool = False,
        pitch: float = 0.0,
        storage: "StorageManager | None" = None,
        progress_callback: Callable[[float, str], None] | None = None,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    ) -> SynthesisResult:
        """Synthesise ``text`` in the voice of ``reference_audio``.

        This is a blocking call — use :meth:`clone_voice_async` from the API.
        """
        # ---- validation -------------------------------------------------
        payload = (text or "").strip()
        if not payload:
            raise TTSValidationError("Text must not be empty.")
        if len(payload) > self.settings.tts_max_text_length:
            raise TTSValidationError(
                f"Text exceeds the {self.settings.tts_max_text_length} character limit.",
                details={"length": len(payload)},
            )

        speaker = Path(reference_audio)
        if not speaker.exists():
            raise TTSValidationError(
                "The reference audio clip used to clone the voice does not exist.",
                details={"reference_audio": str(reference_audio)},
            )

        language_code = normalize_language(language, default=self.settings.tts_default_language)
        speed_value = max(0.25, min(3.0, float(speed or 1.0)))
        pitch_value = max(-12.0, min(12.0, float(pitch or 0.0)))

        # ---- destination ------------------------------------------------
        if output_path:
            destination = Path(output_path)
        else:
            root = storage.outputs_dir if storage else self.settings.outputs_dir
            root.mkdir(parents=True, exist_ok=True)
            from uuid import uuid4

            destination = root / f"tts-{uuid4().hex[:12]}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)

        model = self.load_model()
        device = self.resolve_device()
        chunks = self.chunk_text(payload, max_chunk_chars) or [payload]

        def notify(progress: float, message: str) -> None:
            if progress_callback:
                try:
                    progress_callback(progress, message)
                except Exception:  # noqa: BLE001 - progress must never break synthesis
                    logger.debug("Progress callback failed", exc_info=True)

        notify(5.0, "Loading voice model…")

        staged: list[Path] = []
        scratch = destination.parent / f"{destination.stem}-chunks"
        scratch.mkdir(parents=True, exist_ok=True)

        try:
            with self._inference_lock:
                for index, chunk in enumerate(chunks):
                    notify(
                        10.0 + (index / max(1, len(chunks))) * 75.0,
                        f"Synthesising segment {index + 1}/{len(chunks)}…",
                    )
                    part = scratch / f"part-{index:04d}.wav"
                    self._synthesise_chunk(
                        model=model,
                        text=chunk,
                        speaker=speaker,
                        language=language_code,
                        speed=speed_value,
                        destination=part,
                    )
                    staged.append(part)

            notify(88.0, "Assembling audio…")
            if len(staged) == 1:
                destination.write_bytes(staged[0].read_bytes())
            else:
                self.audio.concat_audio(staged, destination, sample_rate=self.settings.audio_sample_rate)

            if normalize:
                notify(92.0, "Normalising loudness…")
                normalised = destination.with_name(f"{destination.stem}.norm.wav")
                self.audio.normalize_audio(destination, normalised)
                normalised.replace(destination)

            if abs(pitch_value) >= 0.01:
                notify(96.0, f"Shifting pitch {pitch_value:+.1f} semitones…")
                shifted = destination.with_name(f"{destination.stem}.pitch.wav")
                self.audio.shift_pitch(
                    destination,
                    shifted,
                    pitch_value,
                    sample_rate=self.settings.audio_sample_rate,
                )
                shifted.replace(destination)
        finally:
            for part in staged:
                part.unlink(missing_ok=True)
            if scratch.exists():
                for leftover in scratch.glob("*"):
                    leftover.unlink(missing_ok=True)
                scratch.rmdir()

        duration = self.audio.get_duration(destination)
        notify(100.0, "Synthesis complete.")
        logger.info(
            "Synthesised %d chunk(s) -> %s (%.2fs, device=%s)",
            len(chunks), destination.name, duration or 0.0, device,
        )
        return SynthesisResult(
            audio_path=destination,
            duration_seconds=duration,
            language=language_code,
            chunks=len(chunks),
            device=device,
            sample_rate=self.settings.audio_sample_rate,
        )

    def _synthesise_chunk(
        self,
        *,
        model: Any,
        text: str,
        speaker: Path,
        language: str,
        speed: float,
        destination: Path,
    ) -> Path:
        common: dict[str, Any] = {
            "text": text,
            "speaker_wav": str(speaker),
            "language": language,
            "file_path": str(destination),
        }
        try:
            model.tts_to_file(speed=speed, **common)
        except TypeError:
            # Older Coqui releases do not accept `speed` on tts_to_file.
            logger.debug("tts_to_file(speed=…) unsupported; retrying without it.")
            model.tts_to_file(**common)

        if not destination.exists() or destination.stat().st_size == 0:
            raise TTSEngineUnavailable(
                "XTTS returned an empty audio file. Check the reference clip and text."
            )
        return destination

    async def clone_voice_async(self, *args: Any, **kwargs: Any) -> SynthesisResult:
        """Run :meth:`clone_voice` in a worker thread (keeps the loop free)."""
        return await asyncio.to_thread(self.clone_voice, *args, **kwargs)


__all__ = [
    "SynthesisResult",
    "SUPPORTED_LANGUAGES",
    "TTSEngine",
    "TTSEngineUnavailable",
    "TTSValidationError",
    "normalize_language",
]
