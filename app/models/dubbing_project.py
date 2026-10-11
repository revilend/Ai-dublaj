"""Video dubbing projects and their lifecycle status."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from sqlalchemy import Enum as SAEnum
from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.subtitle_segment import SubtitleSegment
    from app.models.voice_profile import VoiceProfile


class DubbingStatus(str, enum.Enum):
    """Lifecycle of a dubbing project."""

    PENDING = "pending"
    EXTRACTING = "extracting"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    SYNTHESIZING = "synthesizing"
    MERGING = "merging"
    COMPLETED = "completed"
    FAILED = "failed"

    @classmethod
    def terminal(cls) -> frozenset["DubbingStatus"]:
        return frozenset({cls.COMPLETED, cls.FAILED})


class DubbingProject(TimestampMixin, Base):
    """One source video travelling through the dubbing pipeline."""

    __tablename__ = "dubbing_projects"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False, index=True)

    # --- media artifacts (paths relative to the data directory) ----------
    video_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    original_audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # Separation outputs: `vocals.wav` (dialogue) and `background.wav` (BGM/FX).
    vocals_audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    background_audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    dubbed_audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    final_video_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # --- pipeline state ---------------------------------------------------
    status: Mapped[DubbingStatus] = mapped_column(
        SAEnum(DubbingStatus, native_enum=False, validate_strings=True, length=24),
        nullable=False,
        default=DubbingStatus.PENDING,
        index=True,
    )
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- languages & voice ------------------------------------------------
    source_language: Mapped[str] = mapped_column(String(16), nullable=False, default="auto")
    target_language: Mapped[str] = mapped_column(String(16), nullable=False, default="en")
    voice_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("voice_profiles.id", ondelete="SET NULL"), nullable=True, index=True
    )
    speaker_reference_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    last_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    segment_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # --- relationships ----------------------------------------------------
    voice_profile: Mapped["VoiceProfile | None"] = relationship(back_populates="projects")
    segments: Mapped[list["SubtitleSegment"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="SubtitleSegment.start_time",
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<DubbingProject id={self.id} title={self.title!r} status={self.status.value}>"
