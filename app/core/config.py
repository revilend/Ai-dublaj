"""Typed, environment-driven application configuration.

Every value can be overridden with an environment variable using the ``VOX_``
prefix (see ``README.md`` for the full list). Configuration is cached so that
importing it repeatedly is free.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ``app/core/config.py`` -> ``app/core`` -> ``app`` -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

ALLOWED_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus", ".webm"}
)
ALLOWED_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg"}
)
ALLOWED_SUBTITLE_EXTENSIONS: frozenset[str] = frozenset({".srt", ".vtt"})

_VALID_LOG_LEVELS = frozenset(
    {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
)


class Settings(BaseSettings):
    """Runtime configuration for the studio backend."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="VOX_",
        case_sensitive=False,
        extra="ignore",
    )

    # --- application -----------------------------------------------------
    app_name: str = "VoxStudio"
    app_description: str = (
        "AI voice cloning & video dubbing studio — fully local, open-source models."
    )
    app_version: str = "0.1.0"
    environment: Literal["development", "production", "test"] = "development"
    debug: bool = True
    log_level: str = "INFO"

    # --- server ----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"

    # --- storage ---------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    database_url: str | None = None
    max_upload_size_mb: int = Field(default=512, gt=0)

    # --- text-to-speech (Coqui XTTS-v2) ----------------------------------
    tts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_device: Literal["auto", "cuda", "cpu"] = "auto"
    tts_default_language: str = "en"
    tts_default_speed: float = Field(default=1.0, ge=0.25, le=3.0)
    tts_max_text_length: int = Field(default=5000, gt=0)
    audio_sample_rate: int = Field(default=24000, gt=0)

    # --- media processing ------------------------------------------------
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    enable_background_separation: bool = True
    background_separation_model: str = "htdemucs"

    # --- optional AI pipelines ------------------------------------------
    translation_model_name: str = "facebook/nllb-200-distilled-600M"
    # `small` is the recommended floor for dubbing; `medium` is more accurate
    # but roughly 4x slower on CPU.
    whisper_model_size: str = "small"

    # --- speaker diarization --------------------------------------------
    # Real diarization requires pyannote.audio and a gated Hugging Face model.
    # Left off by default: without it every segment keeps a null speaker id
    # instead of being guessed at.
    enable_diarization: bool = True
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    hf_token: str | None = None

    # --- dubbing workflow -------------------------------------------------
    dubbing_analyze_on_upload: bool = True
    # How much of a cue's duration a generated clip may be time-stretched to fit.
    dubbing_max_time_stretch: float = Field(default=1.75, ge=1.0, le=4.0)

    # --- background tasks ------------------------------------------------
    task_retention_seconds: int = Field(default=3600, ge=0)
    task_max_history: int = Field(default=500, gt=0)

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = (value or "INFO").strip().upper()
        if level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log level '{value}'. "
                f"Expected one of: {', '.join(sorted(_VALID_LOG_LEVELS))}."
            )
        return level

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand_data_dir(cls, value: object) -> object:
        if isinstance(value, Path):
            return value
        if isinstance(value, str) and value.strip():
            return Path(value).expanduser()
        return value

    @field_validator("database_url", "hf_token", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    # ------------------------------------------------------------------ #
    # Derived paths / helpers
    # ------------------------------------------------------------------ #
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def temp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def has_hf_token(self) -> bool:
        """True when a Hugging Face token is configured (for gated models)."""
        return bool(self.hf_token)

    @property
    def static_dir(self) -> Path:
        return PROJECT_ROOT / "app" / "static"

    @property
    def templates_dir(self) -> Path:
        return PROJECT_ROOT / "app" / "templates"

    @property
    def managed_directories(self) -> tuple[Path, ...]:
        return (
            self.data_dir,
            self.uploads_dir,
            self.voices_dir,
            self.outputs_dir,
            self.temp_dir,
        )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'studio.db').as_posix()}"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def cors_origin_list(self) -> list[str]:
        origins = [item.strip() for item in self.cors_origins.split(",") if item.strip()]
        return origins or ["*"]

    def ensure_directories(self) -> None:
        """Create every directory the application writes to (idempotent)."""
        for directory in self.managed_directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (used by tests)."""
    get_settings.cache_clear()
