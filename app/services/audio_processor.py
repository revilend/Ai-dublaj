"""ffmpeg/ffprobe wrapper used by the dubbing pipeline.

The processor never imports a heavyweight library at module import time: it
shells out to ``ffmpeg`` and therefore boots instantly. Every method raises
:class:`~app.core.exceptions.MediaProcessingError` (HTTP 422) when the media is
unusable and :class:`~app.core.exceptions.ServiceUnavailableError` (HTTP 503)
when the binary itself is missing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.exceptions import MediaProcessingError, ServiceUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)

FFMPEG_TIMEOUT_SECONDS = 1800


@dataclass(slots=True)
class SeparatedStems:
    """Result of vocal/background separation.

    Both paths are ``None`` when separation is unavailable — callers treat that
    as "keep the original audio bed" rather than as an error.
    """

    vocals: Path | None
    background: Path | None

    @property
    def ok(self) -> bool:
        return self.vocals is not None and self.background is not None


@dataclass(slots=True)
class MediaInfo:
    """Normalised result of ``ffprobe``."""

    path: str
    duration: float | None
    has_video: bool
    has_audio: bool
    sample_rate: int | None = None
    channels: int | None = None
    format_name: str | None = None

    @property
    def is_audio_only(self) -> bool:
        return self.has_audio and not self.has_video


class AudioProcessor:
    """High-level media operations expressed as ffmpeg command lines."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._ffmpeg = shutil.which(self.settings.ffmpeg_binary)
        self._ffprobe = shutil.which(self.settings.ffprobe_binary)

    # ------------------------------------------------------------------ #
    # Capabilities
    # ------------------------------------------------------------------ #
    @property
    def ffmpeg_available(self) -> bool:
        return self._ffmpeg is not None

    @property
    def ffprobe_available(self) -> bool:
        return self._ffprobe is not None

    def capability_detail(self) -> str:
        if self.ffmpeg_available and self.ffprobe_available:
            return f"ffmpeg found at {self._ffmpeg}"
        missing = []
        if not self.ffmpeg_available:
            missing.append(self.settings.ffmpeg_binary)
        if not self.ffprobe_available:
            missing.append(self.settings.ffprobe_binary)
        return f"Missing binaries on PATH: {', '.join(missing)}"

    def require_ffmpeg(self) -> str:
        if self._ffmpeg is None:
            raise ServiceUnavailableError(
                f"'{self.settings.ffmpeg_binary}' is not installed or not on PATH. "
                "Install ffmpeg to enable media processing.",
                details={"missing": self.settings.ffmpeg_binary},
            )
        return self._ffmpeg

    def require_ffprobe(self) -> str:
        if self._ffprobe is None:
            raise ServiceUnavailableError(
                f"'{self.settings.ffprobe_binary}' is not installed or not on PATH.",
                details={"missing": self.settings.ffprobe_binary},
            )
        return self._ffprobe

    # ------------------------------------------------------------------ #
    # Low level
    # ------------------------------------------------------------------ #
    def _run(
        self,
        args: list[str],
        *,
        timeout: int = FFMPEG_TIMEOUT_SECONDS,
        action: str = "process media",
    ) -> subprocess.CompletedProcess[str]:
        logger.debug("Running: %s", " ".join(args))
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MediaProcessingError(
                f"Timed out while trying to {action} (>{timeout}s).",
                details={"command": args[0]},
            ) from exc
        except OSError as exc:  # pragma: no cover - binary vanished mid-run
            raise ServiceUnavailableError(f"Could not execute {args[0]}: {exc}") from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip().splitlines()
            tail = "\n".join(stderr[-8:]) if stderr else "unknown ffmpeg error"
            raise MediaProcessingError(
                f"Failed to {action}.",
                details={"exit_code": result.returncode, "stderr": tail},
            )
        return result

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #
    def probe(self, path: str | Path) -> MediaInfo:
        target = Path(path)
        if not target.exists():
            raise MediaProcessingError(f"Media file '{target.name}' does not exist.")
        ffprobe = self.require_ffprobe()
        result = self._run(
            [
                ffprobe, "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(target),
            ],
            action=f"inspect '{target.name}'",
        )
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupt file
            raise MediaProcessingError(
                f"Could not parse media metadata for '{target.name}'."
            ) from exc

        streams = payload.get("streams") or []
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

        duration: float | None = None
        raw_duration = (payload.get("format") or {}).get("duration")
        if raw_duration is not None:
            try:
                duration = float(raw_duration)
            except (TypeError, ValueError):
                duration = None

        sample_rate = None
        channels = None
        if audio:
            try:
                sample_rate = int(audio.get("sample_rate")) if audio.get("sample_rate") else None
            except (TypeError, ValueError):
                sample_rate = None
            channels = audio.get("channels")

        return MediaInfo(
            path=str(target),
            duration=duration,
            has_video=video is not None,
            has_audio=audio is not None,
            sample_rate=sample_rate,
            channels=channels,
            format_name=(payload.get("format") or {}).get("format_name"),
        )

    def get_duration(self, path: str | Path) -> float | None:
        """Duration in seconds, or ``None`` when it cannot be determined."""
        try:
            return self.probe(path).duration
        except (MediaProcessingError, ServiceUnavailableError):
            return None

    # ------------------------------------------------------------------ #
    # Transcoding
    # ------------------------------------------------------------------ #
    def extract_audio(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        sample_rate: int | None = None,
        channels: int = 1,
        codec: str = "pcm_s16le",
    ) -> Path:
        """Extract (and downmix) the audio track of a video/audio file."""
        ffmpeg = self.require_ffmpeg()
        out = Path(destination)
        out.parent.mkdir(parents=True, exist_ok=True)
        src = Path(source)
        if not src.exists():
            raise MediaProcessingError(f"Source file '{src.name}' does not exist.")

        self._run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(src),
                "-vn", "-acodec", codec,
                "-ar", str(sample_rate or self.settings.audio_sample_rate),
                "-ac", str(channels),
                str(out),
            ],
            action=f"extract audio from '{src.name}'",
        )
        return out

    def convert_audio(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        sample_rate: int | None = None,
        channels: int = 1,
    ) -> Path:
        return self.extract_audio(
            source, destination, sample_rate=sample_rate, channels=channels
        )

    def normalize_audio(
        self, source: str | Path, destination: str | Path
    ) -> Path:
        """EBU R128 loudness normalisation (``loudnorm``)."""
        ffmpeg = self.require_ffmpeg()
        out = Path(destination)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(source),
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-acodec", "pcm_s16le",
                str(out),
            ],
            action="normalise audio",
        )
        return out

    def concat_audio(
        self,
        sources: list[str | Path],
        destination: str | Path,
        *,
        sample_rate: int | None = None,
    ) -> Path:
        """Concatenate WAV clips with the ffmpeg concat demuxer."""
        if not sources:
            raise MediaProcessingError("No audio clips were supplied to concatenate.")
        if len(sources) == 1:
            out = Path(destination)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(sources[0]), out)
            return out

        ffmpeg = self.require_ffmpeg()
        out = Path(destination)
        out.parent.mkdir(parents=True, exist_ok=True)
        list_file = out.parent / f"{out.stem}.concat.txt"
        list_file.write_text(
            "\n".join(f"file '{Path(item).resolve().as_posix()}'" for item in sources) + "\n",
            encoding="utf-8",
        )
        try:
            self._run(
                [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", str(list_file),
                    "-acodec", "pcm_s16le",
                    "-ar", str(sample_rate or self.settings.audio_sample_rate),
                    "-ac", "1",
                    str(out),
                ],
                action="concatenate audio clips",
            )
        finally:
            list_file.unlink(missing_ok=True)
        return out

    def generate_silence(
        self,
        duration_seconds: float,
        destination: str | Path,
        *,
        sample_rate: int | None = None,
    ) -> Path:
        """Render a mono WAV of silence (used to pad the dub timeline)."""
        ffmpeg = self.require_ffmpeg()
        out = Path(destination)
        out.parent.mkdir(parents=True, exist_ok=True)
        rate = sample_rate or self.settings.audio_sample_rate
        self._run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi",
                "-i", f"anullsrc=r={rate}:cl=mono",
                "-t", f"{max(0.01, duration_seconds):.3f}",
                "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(rate),
                str(out),
            ],
            action="generate a silent audio pad",
        )
        return out

    def shift_pitch(
        self,
        source: str | Path,
        destination: str | Path,
        semitones: float,
        *,
        sample_rate: int | None = None,
    ) -> Path:
        """Shift pitch by ``semitones`` while preserving the original duration.

        Uses the classic ffmpeg recipe — reinterpret the sample rate
        (``asetrate``) to move pitch, resample back, then undo the resulting
        duration change with ``atempo``. No extra filter libraries required.
        """
        ffmpeg = self.require_ffmpeg()
        out = Path(destination)
        out.parent.mkdir(parents=True, exist_ok=True)
        src = Path(source)

        shift = max(-12.0, min(12.0, float(semitones or 0.0)))
        if abs(shift) < 0.01:
            shutil.copyfile(src, out)
            return out

        rate = int(sample_rate or self.settings.audio_sample_rate)
        if not sample_rate:
            probed = self.probe(src).sample_rate
            if probed:
                rate = int(probed)

        factor = 2.0 ** (shift / 12.0)
        filter_chain = (
            f"asetrate={int(rate * factor)},"
            f"aresample={rate},"
            f"{self._atempo_chain(1.0 / factor)}"
        )
        self._run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(src),
                "-filter:a", filter_chain,
                "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(rate),
                str(out),
            ],
            action="shift the pitch of a clip",
        )
        return out

    def mix_audio(
        self,
        primary: str | Path,
        background: str | Path,
        destination: str | Path,
        *,
        primary_gain_db: float = 0.0,
        background_gain_db: float = -6.0,
    ) -> Path:
        """Mix voice over background music/ambience (``amix``)."""
        ffmpeg = self.require_ffmpeg()
        out = Path(destination)
        out.parent.mkdir(parents=True, exist_ok=True)
        filter_graph = (
            f"[0:a]volume={primary_gain_db}dB[v];"
            f"[1:a]volume={background_gain_db}dB[b];"
            f"[v][b]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[a]"
        )
        self._run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(primary), "-i", str(background),
                "-filter_complex", filter_graph,
                "-map", "[a]", "-acodec", "pcm_s16le", "-ac", "1",
                "-ar", str(self.settings.audio_sample_rate),
                str(out),
            ],
            action="mix dubbed audio with background",
        )
        return out

    def fit_to_duration(
        self,
        source: str | Path,
        destination: str | Path,
        target_seconds: float,
        *,
        tolerance: float = 0.05,
    ) -> Path:
        """Time-stretch then pad/trim a clip so it matches a cue exactly."""
        if target_seconds <= 0:
            raise MediaProcessingError("Target duration must be greater than zero.")

        ffmpeg = self.require_ffmpeg()
        out = Path(destination)
        out.parent.mkdir(parents=True, exist_ok=True)
        src = Path(source)
        duration = self.get_duration(src)

        staged = src
        if duration and duration > 0:
            ratio = duration / target_seconds
            if abs(ratio - 1.0) > tolerance:
                stretched = out.with_name(f"{out.stem}.stretch.wav")
                self._run(
                    [
                        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(src),
                        "-filter:a", self._atempo_chain(ratio),
                        "-acodec", "pcm_s16le",
                        str(stretched),
                    ],
                    action="time-stretch a dubbed segment",
                )
                staged = stretched

        try:
            # `apad` + `-t` both extends short clips and trims long ones.
            self._run(
                [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(staged),
                    "-af", "apad",
                    "-t", f"{target_seconds:.3f}",
                    "-acodec", "pcm_s16le", "-ac", "1",
                    "-ar", str(self.settings.audio_sample_rate),
                    str(out),
                ],
                action="fit a dubbed segment to its cue",
            )
        finally:
            if staged != src:
                Path(staged).unlink(missing_ok=True)
        return out

    @staticmethod
    def pitch_ratio(semitones: float) -> float:
        """Frequency multiplier for a semitone delta (exposed for tests)."""
        return 2.0 ** (max(-12.0, min(12.0, float(semitones or 0.0))) / 12.0)

    @staticmethod
    def _atempo_chain(ratio: float) -> str:
        """ATempo only accepts 0.5–2.0, so chain filters for larger factors."""
        ratio = max(0.1, min(10.0, ratio))
        factors: list[float] = []
        remaining = ratio
        while remaining < 0.5:
            factors.append(0.5)
            remaining /= 0.5
        while remaining > 2.0:
            factors.append(2.0)
            remaining /= 2.0
        factors.append(remaining)
        return ",".join(f"atempo={factor:.4f}" for factor in factors)

    def mux_video(
        self,
        video: str | Path,
        audio: str | Path,
        destination: str | Path,
        *,
        audio_codec: str = "aac",
        audio_bitrate: str = "192k",
    ) -> Path:
        """Replace a video's audio track with the dubbed one (stream copy video)."""
        ffmpeg = self.require_ffmpeg()
        out = Path(destination)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video), "-i", str(audio),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", audio_codec, "-b:a", audio_bitrate,
                "-shortest", str(out),
            ],
            action="mux the dubbed audio back into the video",
        )
        return out

    # ------------------------------------------------------------------ #
    # Background / vocal separation (optional Demucs)
    # ------------------------------------------------------------------ #
    @property
    def background_separation_available(self) -> bool:
        if not self.settings.enable_background_separation:
            return False
        try:
            import demucs  # noqa: F401
        except Exception:  # noqa: BLE001 - any import failure means "unavailable"
            return False
        return True

    def separate_stems(
        self, source: str | Path, output_dir: str | Path | None = None
    ) -> SeparatedStems:
        """Split audio into ``vocals.wav`` (dialogue) and ``background.wav`` (BGM/FX).

        Uses Demucs' two-stem mode when installed. Demucs writes its own
        ``vocals.wav`` / ``no_vocals.wav`` names, so the stems are normalised
        into the destination directory under the names the dubbing pipeline
        refers to. Missing Demucs yields empty stems, never an exception.
        """
        empty = SeparatedStems(vocals=None, background=None)

        if not self.settings.enable_background_separation:
            logger.info("Background separation disabled by configuration.")
            return empty
        if not self.background_separation_available:
            logger.info("Demucs is not installed — skipping vocal separation.")
            return empty

        src = Path(source)
        destination = Path(output_dir) if output_dir else src.parent / "separated"
        destination.mkdir(parents=True, exist_ok=True)
        workdir = destination / "_demucs"

        command = [
            sys.executable, "-m", "demucs",
            "--two-stems=vocals",
            "-n", self.settings.background_separation_model,
            "--out", str(workdir),
            str(src),
        ]
        try:
            self._run(command, action="separate vocals from background", timeout=3600)
        except (MediaProcessingError, ServiceUnavailableError) as exc:
            logger.warning("Vocal separation failed (%s); continuing without it.", exc)
            return empty

        produced = workdir / self.settings.background_separation_model / src.stem
        vocals_src = produced / "vocals.wav"
        background_src = produced / "no_vocals.wav"
        if not vocals_src.exists() or not background_src.exists():
            logger.warning("Demucs produced no stems for %s (looked in %s).", src.name, produced)
            return empty

        vocals = destination / "vocals.wav"
        background = destination / "background.wav"
        shutil.copyfile(vocals_src, vocals)
        shutil.copyfile(background_src, background)
        shutil.rmtree(workdir, ignore_errors=True)

        logger.info("Separated stems for %s -> vocals.wav / background.wav", src.name)
        return SeparatedStems(vocals=vocals, background=background)


__all__ = ["AudioProcessor", "MediaInfo", "SeparatedStems"]
