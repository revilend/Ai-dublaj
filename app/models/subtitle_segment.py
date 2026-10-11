"""Timed subtitle/caption segments belonging to a dubbing project."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.dubbing_project import DubbingProject
    from app.models.voice_profile import VoiceProfile


class SubtitleSegment(TimestampMixin, Base):
    """A single timed cue with original, translated and dubbed representations."""

    __tablename__ = "subtitle_segments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("dubbing_projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    start_time: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    end_time: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    original_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    translated_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    speaker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dubbed_audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Optional per-segment voice: overrides the project's voice profile, which
    # is what makes multi-speaker dubbing possible. NULL means "use the
    # project default".
    voice_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("voice_profiles.id", ondelete="SET NULL"), nullable=True
    )

    project: Mapped["DubbingProject"] = relationship(back_populates="segments")
    voice_profile: Mapped["VoiceProfile | None"] = relationship(lazy="joined")

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end_time) - float(self.start_time))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<SubtitleSegment id={self.id} project={self.project_id} "
            f"{self.start_time:.2f}-{self.end_time:.2f}s>"
        )
