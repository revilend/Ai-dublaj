"""The dubbing engine.

The pipeline is reported to clients as four discrete steps, using the exact
labels the studio UI displays::

    Step 1/4: Separating Music    extract audio -> Demucs -> vocals.wav + background.wav
    Step 2/4: Transcribing        faster-whisper -> timed segments (+ diarization)
    Step 3/4: Generating Voices   NLLB translation -> XTTS-v2 -> time-stretch to fit
    Step 4/4: Rendering Video     rebuild the timeline -> mix background -> FFmpeg mux

Steps 1–2 are the *analysis* phase (``POST /api/dubbing/upload``); steps 3–4 are
the *render* phase (``POST /api/dubbing/render/{project_id}``). A render run
re-enters steps 1–2 only when their artifacts are missing, which makes the whole
pipeline resumable.

Progress is published through the :class:`TaskContext` (WebSocket + polling),
and every failure leaves the project in a ``failed`` state carrying an
actionable message instead of a traceback.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.core.database import session_scope
from app.core.exceptions import AppError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.models.dubbing_project import DubbingProject, DubbingStatus
from app.models.subtitle_segment import SubtitleSegment
from app.models.voice_profile import VoiceProfile
from app.schemas.dubbing import RenderRequest
from app.services.audio_processor import AudioProcessor
from app.services.diarization_service import DiarizationService
from app.services.storage import StorageManager
from app.services.task_manager import TaskContext
from app.services.transcription_service import TranscriptionService
from app.services.translation_service import TranslationService
from app.services.tts_engine import TTSEngine, normalize_language

logger = get_logger(__name__)

STEP_TOTAL = 4

STEP_MESSAGES: dict[int, str] = {
    1: "Step 1/4: Separating Music",
    2: "Step 2/4: Transcribing",
    3: "Step 3/4: Generating Voices",
    4: "Step 4/4: Rendering Video",
}

# Progress window (start, end) reserved for each step, in percent.
STEP_PROGRESS: dict[int, tuple[float, float]] = {
    1: (2.0, 20.0),
    2: (22.0, 45.0),
    3: (48.0, 88.0),
    4: (90.0, 99.0),
}

STEP_STATUS: dict[int, DubbingStatus] = {
    1: DubbingStatus.EXTRACTING,
    2: DubbingStatus.TRANSCRIBING,
    3: DubbingStatus.SYNTHESIZING,
    4: DubbingStatus.MERGING,
}

# Statuses that only make sense *while* the pipeline is running. An editor-side
# action must never leave a project parked in one of these.
TRANSIENT_STATUSES: frozenset[DubbingStatus] = frozenset(
    {
        DubbingStatus.EXTRACTING,
        DubbingStatus.TRANSCRIBING,
        DubbingStatus.TRANSLATING,
        DubbingStatus.SYNTHESIZING,
        DubbingStatus.MERGING,
    }
)


class DubbingService:
    """Orchestrates the dubbing pipeline end to end."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        storage: StorageManager | None = None,
        audio: AudioProcessor | None = None,
        tts: TTSEngine | None = None,
        transcriber: TranscriptionService | None = None,
        translator: TranslationService | None = None,
        diarizer: DiarizationService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.storage = storage or StorageManager(self.settings)
        self.audio = audio or AudioProcessor(self.settings)
        self.tts = tts or TTSEngine(self.settings, audio_processor=self.audio)
        self.transcriber = transcriber or TranscriptionService(self.settings)
        self.translator = translator or TranslationService(self.settings)
        self.diarizer = diarizer or DiarizationService(self.settings)

    # ------------------------------------------------------------------ #
    # Public entrypoints
    # ------------------------------------------------------------------ #
    async def analyze(
        self, ctx: TaskContext, project_id: int, options: RenderRequest | None = None
    ) -> dict[str, object]:
        """Run steps 1–2 only (extract, separate, transcribe)."""
        return await self.run(ctx, project_id, options, stop_after_analysis=True)

    async def render(
        self, ctx: TaskContext, project_id: int, options: RenderRequest | None = None
    ) -> dict[str, object]:
        """Run the whole pipeline, skipping steps whose artifacts exist."""
        return await self.run(ctx, project_id, options, stop_after_analysis=False)

    async def preview_segment(
        self, ctx: TaskContext, segment_id: int, options: RenderRequest | None = None
    ) -> dict[str, object]:
        """Synthesise a *single* line and store it as that segment's audio.

        This backs the editor's "Preview line" button: it clones just this cue
        with whichever voice profile the line resolves to, so a translation can
        be auditioned (and corrected) long before a full render. The produced
        clip is written to the segment's ``dubbed_audio_path``, so a later render
        simply overwrites it.
        """
        options = options or RenderRequest()
        segment = self._load_segment(segment_id)
        project = self._load_project(segment.project_id)

        text = (segment.translated_text or segment.original_text or "").strip()
        if not text:
            raise ValidationError(
                f"Segment {segment_id} has no text to synthesise.",
                details={"segment_id": segment_id},
            )

        language = normalize_language(
            options.target_language or project.target_language or "en"
        )
        references = self._reference_cache(project.id, options)
        override = (
            references.get(options.voice_profile_id) if options.voice_profile_id else None
        )
        reference = (
            override
            or references.get(segment.voice_profile_id)
            or references.get(project.voice_profile_id)
        )
        if reference is None:
            raise ValidationError(
                "Assign a voice profile to this line (or set a project voice) "
                "before previewing it.",
                details={"segment_id": segment_id, "voice_profile_id": segment.voice_profile_id},
            )

        meta = {
            "project_id": project.id,
            "segment_id": segment_id,
            "step_total": STEP_TOTAL,
            "step": 3,
            "phase": "preview",
        }
        ctx.report(10.0, f"Cloning line {segment.order_index + 1}", metadata=meta)

        workdir = self.storage.outputs_dir / f"project-{project.id}"
        workdir.mkdir(parents=True, exist_ok=True)
        scratch = self.storage.stage_dir(f"preview-{segment_id}")
        fitted = workdir / f"segment-{segment_id:04d}.wav"
        try:
            raw = scratch / f"line-{segment_id:04d}.wav"
            result = await self.tts.clone_voice_async(
                text,
                reference,
                language=language,
                speed=self.settings.tts_default_speed,
                output_path=raw,
                progress_callback=lambda value, detail: ctx.report(
                    10.0 + (value / 100.0) * 70.0, detail, metadata=meta
                ),
            )

            target = float(segment.duration or 0.0)
            if target > 0:
                ctx.report(85.0, "Fitting the line to its cue", metadata=meta)
                await asyncio.to_thread(
                    self.audio.fit_to_duration, result.audio_path, fitted, target
                )
            else:
                # No cue span to fit: keep the raw clone instead of failing.
                await asyncio.to_thread(shutil.copyfile, result.audio_path, fitted)
        finally:
            self.storage.delete(scratch)

        with session_scope() as db:
            row = db.get(SubtitleSegment, segment_id)
            if row is not None:
                row.dubbed_audio_path = self.storage.relative(fitted)

        duration = self.audio.get_duration(fitted)
        logger.info("Previewed segment %s -> %s", segment_id, fitted.name)
        ctx.report(100.0, "Preview ready", metadata=meta)
        return {
            "segment_id": segment_id,
            "project_id": project.id,
            "dubbed_audio_path": self.storage.relative(fitted),
            "dubbed_audio_url": self.storage.url_for(fitted),
            "duration_seconds": duration,
        }

    async def translate_segments(
        self, ctx: TaskContext, project_id: int, options: RenderRequest | None = None
    ) -> dict[str, object]:
        """Auto-translate every segment **without** synthesising any audio.

        Backs the editor's "Auto-translate" action so the translation column can
        be filled (and then hand-edited) before a render.
        """
        options = options or RenderRequest()
        segments = self._load_segments(project_id)
        if not segments:
            raise ValidationError(
                "The project has no segments to translate. Upload an SRT/VTT or "
                "run the analysis phase first.",
                details={"project_id": project_id},
            )
        project = self._load_project(project_id)
        meta = {"project_id": project_id, "step_total": STEP_TOTAL, "step": 3, "phase": "translation"}

        def report(progress: float, detail: str) -> None:
            ctx.report(progress, detail, metadata=meta)

        requested = RenderRequest(
            translate=True,
            target_language=options.target_language,
            voice_profile_id=options.voice_profile_id,
        )
        report(10.0, "Translating the subtitle track")
        try:
            await self._translate(project_id, segments, requested, report)
        finally:
            # `_translate` parks the project in `translating`. This is an editor
            # action, not a pipeline stage, so always restore a settled status —
            # otherwise a failure would strand the project on "translating"
            # forever and the UI could never leave that state.
            settled = (
                DubbingStatus.PENDING
                if project.status in TRANSIENT_STATUSES
                else project.status
            )
            self._update_project(project_id, status=settled, progress=0.0)

        updated = self._load_segments(project_id)
        filled = sum(1 for row in updated if (row.translated_text or "").strip())
        if not filled:
            raise ValidationError(
                "Nothing was translated — the source and target languages match, "
                "or NLLB does not support that pair. Edit the translation by hand.",
                details={
                    "project_id": project_id,
                    "source_language": project.source_language,
                    "target_language": requested.target_language or project.target_language,
                },
            )

        report(100.0, f"Translated {filled} segment(s)")
        return {"project_id": project_id, "translated": filled, "segments": len(updated)}

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    async def run(
        self,
        ctx: TaskContext,
        project_id: int,
        options: RenderRequest | None = None,
        *,
        stop_after_analysis: bool = False,
    ) -> dict[str, object]:
        options = options or RenderRequest()
        self._load_project(project_id)  # fail fast (and visibly) if it is gone

        workdir = self.storage.outputs_dir / f"project-{project_id}"
        workdir.mkdir(parents=True, exist_ok=True)
        scratch = self.storage.stage_dir(f"dub-{project_id}")

        step = 1
        phase = "preparing"

        def report(progress: float, detail: str) -> None:
            """Publish progress under the current step's label."""
            ctx.report(
                progress,
                STEP_MESSAGES[step],
                metadata={
                    "step": step,
                    "step_total": STEP_TOTAL,
                    "phase": phase,
                    "detail": detail,
                    "project_id": project_id,
                },
            )

        def enter(next_step: int, progress: float, detail: str) -> None:
            nonlocal step, phase
            step = next_step
            phase = STEP_STATUS[next_step].value
            self._update_project(project_id, status=STEP_STATUS[next_step], progress=progress)
            report(progress, detail)

        try:
            self.audio.require_ffmpeg()

            # ---------------- Step 1/4 — Separating Music ---------------- #
            enter(1, STEP_PROGRESS[1][0], "Extracting the original audio track")
            source_audio = await self._extract_audio(project_id, workdir, report)

            vocals, background = await self._separate(project_id, source_audio, workdir, report)
            speech_audio = vocals or source_audio

            # ---------------- Step 2/4 — Transcribing -------------------- #
            enter(2, STEP_PROGRESS[2][0], "Transcribing speech with faster-whisper")
            segments = await self._transcribe(project_id, speech_audio, report)
            if not segments:
                raise ValidationError(
                    "No speech was found in the video. Upload an SRT/VTT with the "
                    "dialogue, or check that the audio track is not silent.",
                    details={"project_id": project_id},
                )

            if stop_after_analysis:
                self._update_project(
                    project_id,
                    status=DubbingStatus.PENDING,
                    progress=0.0,
                    segment_count=len(segments),
                    error_message=None,
                )
                report(STEP_PROGRESS[2][1], f"Analysis complete — {len(segments)} segment(s)")
                return {
                    "project_id": project_id,
                    "stage": "analysis",
                    "segments": len(segments),
                    "vocals_audio_url": self.storage.url_for(vocals),
                    "background_audio_url": self.storage.url_for(background),
                }

            # ---------------- Step 3/4 — Generating Voices --------------- #
            enter(3, STEP_PROGRESS[3][0], "Preparing translations and voices")
            await self._translate(project_id, segments, options, report)
            dubbed_parts = await self._synthesise(project_id, scratch, options, report)

            # ---------------- Step 4/4 — Rendering Video ----------------- #
            enter(4, STEP_PROGRESS[4][0], "Rebuilding the audio timeline")
            timeline = await asyncio.to_thread(self._build_timeline, dubbed_parts, workdir, report)

            if background is not None and options.mix_background:
                report(STEP_PROGRESS[4][0] + 3.0, "Mixing the original background bed back in")
                mixed = workdir / "dubbed-mix.wav"
                timeline = await asyncio.to_thread(
                    self.audio.mix_audio, timeline, background, mixed
                )

            self._update_project(
                project_id,
                dubbed_audio_path=self.storage.relative(timeline),
                progress=95.0,
            )

            report(96.0, "Exporting the dubbed video with FFmpeg")
            final_video = await self._mux(project_id, timeline, workdir)

            duration = self.audio.get_duration(timeline)
            self._update_project(
                project_id,
                status=DubbingStatus.COMPLETED,
                progress=100.0,
                error_message=None,
                final_video_path=self.storage.relative(final_video) if final_video else None,
                dubbed_audio_path=self.storage.relative(timeline),
                duration_seconds=duration,
            )
            ctx.report(100.0, STEP_MESSAGES[4], metadata={"step": 4, "step_total": 4, "phase": "completed"})
            return {
                "project_id": project_id,
                "stage": "render",
                "final_video_path": self.storage.relative(final_video) if final_video else None,
                "final_video_url": self.storage.url_for(final_video) if final_video else None,
                "dubbed_audio_path": self.storage.relative(timeline),
                "dubbed_audio_url": self.storage.url_for(timeline),
                "duration_seconds": duration,
                "segments": len(dubbed_parts),
            }

        except AppError as exc:
            logger.warning("Dubbing project %s failed: %s", project_id, exc.message)
            self._update_project(
                project_id,
                status=DubbingStatus.FAILED,
                error_message=f"{exc.code}: {exc.message}",
                progress=ctx.record.progress,
            )
            raise
        except Exception as exc:  # noqa: BLE001 - unexpected failures still mark the project
            logger.exception("Dubbing project %s crashed", project_id)
            self._update_project(
                project_id,
                status=DubbingStatus.FAILED,
                error_message=str(exc),
                progress=ctx.record.progress,
            )
            raise
        finally:
            self.storage.delete(scratch)

    # ------------------------------------------------------------------ #
    # Step 1 — extract + separate
    # ------------------------------------------------------------------ #
    async def _extract_audio(
        self, project_id: int, workdir: Path, report: Callable[[float, str], None]
    ) -> Path:
        project = self._load_project(project_id)
        if not project.video_path:
            existing = self.storage.absolute(project.original_audio_path)
            if existing is None or not existing.exists():
                raise ValidationError(
                    "The project has neither a video nor an extracted audio track."
                )
            return existing

        video = self.storage.absolute(project.video_path)
        if video is None or not video.exists():
            raise ValidationError("The uploaded video file could not be found on disk.")

        report(STEP_PROGRESS[1][0] + 3.0, "Extracting the original audio track with FFmpeg")
        destination = workdir / "original.wav"
        await asyncio.to_thread(
            self.audio.extract_audio,
            video,
            destination,
            sample_rate=self.settings.audio_sample_rate,
            channels=1,
        )
        self._update_project(
            project_id,
            original_audio_path=self.storage.relative(destination),
            progress=STEP_PROGRESS[1][0] + 4.0,
        )
        return destination

    async def _separate(
        self,
        project_id: int,
        source_audio: Path,
        workdir: Path,
        report: Callable[[float, str], None],
    ) -> tuple[Path | None, Path | None]:
        if not self.audio.background_separation_available:
            report(
                STEP_PROGRESS[1][1],
                "Demucs unavailable — using the original audio for both dialogue and bed",
            )
            return None, None

        report(STEP_PROGRESS[1][0] + 8.0, "Splitting vocals from music with Demucs")
        stems = await asyncio.to_thread(self.audio.separate_stems, source_audio, workdir)

        self._update_project(
            project_id,
            vocals_audio_path=self.storage.relative(stems.vocals) if stems.vocals else None,
            background_audio_path=(
                self.storage.relative(stems.background) if stems.background else None
            ),
            progress=STEP_PROGRESS[1][1],
        )
        report(
            STEP_PROGRESS[1][1],
            "vocals.wav + background.wav ready" if stems.ok else "Separation skipped",
        )
        return stems.vocals, stems.background

    # ------------------------------------------------------------------ #
    # Step 2 — transcribe (+ diarize)
    # ------------------------------------------------------------------ #
    async def _transcribe(
        self, project_id: int, speech_audio: Path, report: Callable[[float, str], None]
    ) -> list[SubtitleSegment]:
        existing = self._load_segments(project_id)
        if existing and any(segment.original_text.strip() for segment in existing):
            report(
                STEP_PROGRESS[2][1],
                f"Using {len(existing)} existing subtitle segment(s)",
            )
            return existing

        def on_progress(value: float, detail: str) -> None:
            span = STEP_PROGRESS[2]
            report(span[0] + (value / 100.0) * (span[1] - span[0]), detail)

        transcript = await self.transcriber.transcribe_async(
            speech_audio,
            language=None,
            progress_callback=on_progress,
        )
        if not transcript:
            return []

        spans = [(item.start, item.end) for item in transcript]
        speakers = await self.diarizer.assign_speakers_async(speech_audio, spans)

        with session_scope() as db:
            db.query(SubtitleSegment).filter(SubtitleSegment.project_id == project_id).delete()
            for index, item in enumerate(transcript):
                db.add(
                    SubtitleSegment(
                        project_id=project_id,
                        start_time=item.start,
                        end_time=item.end,
                        order_index=index,
                        original_text=item.text,
                        speaker_id=speakers[index] if index < len(speakers) else None,
                    )
                )
            project = db.get(DubbingProject, project_id)
            if project is not None:
                project.segment_count = len(transcript)

        labelled = sum(1 for speaker in speakers if speaker)
        report(
            STEP_PROGRESS[2][1],
            f"Transcribed {len(transcript)} segment(s)"
            + (f", {labelled} speaker-labelled" if labelled else ", diarization unavailable"),
        )
        return self._load_segments(project_id)

    # ------------------------------------------------------------------ #
    # Step 3 — translate + synthesise + time-fit
    # ------------------------------------------------------------------ #
    async def _translate(
        self,
        project_id: int,
        segments: list[SubtitleSegment],
        options: RenderRequest,
        report: Callable[[float, str], None],
    ) -> None:
        project = self._load_project(project_id)
        target = (options.target_language or project.target_language or "en").lower()
        source = (project.source_language or "auto").lower()

        if not options.translate or source in ("auto", target):
            return
        if not self.translator.supports(source) or not self.translator.supports(target):
            report(STEP_PROGRESS[3][0] + 1.0, f"Translation skipped — unsupported pair {source}→{target}")
            return

        report(STEP_PROGRESS[3][0] + 1.0, f"Translating {source} → {target} with NLLB")
        self._update_project(project_id, status=DubbingStatus.TRANSLATING)

        texts = [segment.original_text for segment in segments]
        translated = await self.translator.translate_batch_async(
            texts, source_language=source, target_language=target
        )
        with session_scope() as db:
            for segment, value in zip(segments, translated, strict=False):
                row = db.get(SubtitleSegment, segment.id)
                if row is not None:
                    row.translated_text = value
        self._update_project(project_id, target_language=target)

    async def _synthesise(
        self,
        project_id: int,
        scratch: Path,
        options: RenderRequest,
        report: Callable[[float, str], None],
    ) -> list[Path]:
        segments = self._load_segments(project_id)
        if not segments:
            raise ValidationError(
                "The project has no subtitle segments to synthesise. Upload an SRT/VTT first."
            )

        project = self._load_project(project_id)
        default_language = normalize_language(
            options.target_language or project.target_language or "en"
        )
        references = self._reference_cache(project_id, options)
        default_reference = references.get(project.voice_profile_id) or references.get(None)
        # An explicit per-render profile re-dubs every segment with that voice.
        override_reference = (
            references.get(options.voice_profile_id) if options.voice_profile_id else None
        )

        def reference_for(segment: SubtitleSegment) -> Path | None:
            if override_reference is not None:
                return override_reference
            return references.get(segment.voice_profile_id) or default_reference

        speakable = [
            segment
            for segment in segments
            if (segment.translated_text or segment.original_text or "").strip()
        ]
        if not speakable:
            raise ValidationError("No segment contained text to synthesise.")
        if not any(reference_for(segment) for segment in speakable):
            raise ValidationError(
                "No voice reference was provided. Save a voice profile (or upload a "
                "reference clip) before rendering.",
                details={"voice_profile_id": project.voice_profile_id},
            )

        span = STEP_PROGRESS[3]
        dubbed: list[Path] = []
        total = len(segments)

        for index, segment in enumerate(segments):
            report(span[0] + 2.0, f"Preparing segment {index + 1}/{total}")

            text = (segment.translated_text or segment.original_text or "").strip()
            if not text:
                continue

            reference = reference_for(segment)
            if reference is None:
                raise ValidationError(
                    f"Segment {segment.id} references voice profile "
                    f"{segment.voice_profile_id}, which has no usable reference clip."
                )

            progress = span[0] + ((index + 0.5) / total) * (span[1] - span[0])
            report(progress, f"Cloning voice for segment {index + 1}/{total}")

            raw = scratch / f"seg-{index:04d}.wav"
            result = await self.tts.clone_voice_async(
                text,
                reference,
                language=default_language,
                speed=self.settings.tts_default_speed,
                output_path=raw,
                progress_callback=lambda value, detail, base=progress: report(
                    min(base + value / 60.0, span[1]), detail
                ),
            )

            fitted = await self._fit_segment(segment, result.audio_path, scratch, index, report)
            with session_scope() as db:
                row = db.get(SubtitleSegment, segment.id)
                if row is not None:
                    row.dubbed_audio_path = self.storage.relative(fitted)
            dubbed.append(fitted)

        if not dubbed:
            raise ValidationError("No segment contained text to synthesise.")

        report(span[1], f"Generated {len(dubbed)} dubbed clip(s)")
        return dubbed

    async def _fit_segment(
        self,
        segment: SubtitleSegment,
        clip: Path,
        scratch: Path,
        index: int,
        report: Callable[[float, str], None],
    ) -> Path:
        """Time-stretch/pad a clip so it matches the cue's original timestamps."""
        target = float(segment.duration or 0.0)
        natural = self.audio.get_duration(clip)
        if target <= 0:
            return clip

        if natural and natural > 0:
            ratio = natural / target
            if ratio > self.settings.dubbing_max_time_stretch:
                logger.warning(
                    "Segment %s is %.2fx too long for its %.2fs cue; the dub will sound fast. "
                    "Consider shortening the translation.",
                    segment.id,
                    ratio,
                    target,
                )
                report(
                    STEP_PROGRESS[3][0] + 1.0,
                    f"Segment {index + 1}: text is {ratio:.1f}x too long for its cue",
                )

        fitted = scratch / f"fitted-{index:04d}.wav"
        await asyncio.to_thread(self.audio.fit_to_duration, clip, fitted, target)
        return fitted

    # ------------------------------------------------------------------ #
    # Step 4 — rebuild, mix, mux
    # ------------------------------------------------------------------ #
    def _build_timeline(
        self, clips: list[Path], workdir: Path, report: Callable[[float, str], None]
    ) -> Path:
        """Place every clip at its cue start time, padding gaps with silence."""
        starts = self._segment_start_by_clip(clips)

        ordered = sorted(clips, key=lambda path: starts.get(path.name, 0.0))
        timeline_parts: list[Path] = []
        cursor = 0.0

        for index, clip in enumerate(ordered):
            start = starts.get(clip.name, cursor)
            gap = start - cursor
            if gap > 0.05:
                timeline_parts.append(
                    self.audio.generate_silence(gap, workdir / f"gap-{index:04d}.wav")
                )
            timeline_parts.append(clip)
            cursor = start + (self.audio.get_duration(clip) or 0.0)

        destination = workdir / "dubbed.wav"
        report(STEP_PROGRESS[4][0] + 2.0, "Concatenating the dubbed timeline")
        return self.audio.concat_audio(
            timeline_parts, destination, sample_rate=self.settings.audio_sample_rate
        )

    async def _mux(self, project_id: int, timeline: Path, workdir: Path) -> Path | None:
        project = self._load_project(project_id)
        if not project.video_path:
            return None

        video = self.storage.absolute(project.video_path)
        if video is None or not video.exists():
            return None

        final_video = workdir / "final.mp4"
        await asyncio.to_thread(self.audio.mux_video, video, timeline, final_video)
        return final_video

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def _load_project(self, project_id: int) -> DubbingProject:
        with session_scope() as db:
            project = db.get(DubbingProject, project_id)
            if project is None:
                raise NotFoundError(
                    f"Dubbing project {project_id} does not exist.",
                    details={"project_id": project_id},
                )
            db.expunge(project)
            return project

    def _load_segment(self, segment_id: int) -> SubtitleSegment:
        with session_scope() as db:
            segment = db.get(SubtitleSegment, segment_id)
            if segment is None:
                raise NotFoundError(
                    f"Subtitle segment {segment_id} does not exist.",
                    details={"segment_id": segment_id},
                )
            db.expunge(segment)
            return segment

    def _load_segments(self, project_id: int) -> list[SubtitleSegment]:
        with session_scope() as db:
            rows = list(
                db.scalars(
                    select(SubtitleSegment)
                    .where(SubtitleSegment.project_id == project_id)
                    .order_by(SubtitleSegment.start_time)
                )
            )
            for row in rows:
                db.expunge(row)
            return rows

    def _segment_start_by_clip(self, clips: list[Path]) -> dict[str, float]:
        names = {clip.name for clip in clips}
        starts: dict[str, float] = {}
        with session_scope() as db:
            for row in db.scalars(select(SubtitleSegment)):
                name = Path(row.dubbed_audio_path or "").name
                if name in names:
                    starts[name] = row.start_time
        return starts

    def _reference_cache(
        self, project_id: int, options: RenderRequest
    ) -> dict[int | None, Path]:
        """Resolve every voice profile this render needs, once per profile."""
        requested: set[int | None] = {None}
        with session_scope() as db:
            project = db.get(DubbingProject, project_id)
            if project is None:
                raise NotFoundError(f"Dubbing project {project_id} does not exist.")
            requested.add(project.voice_profile_id)
            if options.voice_profile_id:
                project.voice_profile_id = options.voice_profile_id
                requested.add(options.voice_profile_id)
            for row in db.scalars(
                select(SubtitleSegment).where(SubtitleSegment.project_id == project_id)
            ):
                requested.add(row.voice_profile_id)

        references: dict[int | None, Path] = {}

        with session_scope() as db:
            for profile_id in requested:
                if profile_id is None:
                    continue
                profile = db.get(VoiceProfile, profile_id)
                if profile is None:
                    raise NotFoundError(
                        f"Voice profile {profile_id} does not exist.",
                        details={"voice_profile_id": profile_id},
                    )
                path = self.storage.absolute(profile.sample_audio_path)
                if path is None or not path.exists():
                    raise ValidationError(
                        f"The reference clip for voice profile '{profile.name}' is missing on disk."
                    )
                references[profile_id] = path

        if options.speaker_reference_path:
            explicit = self.storage.absolute(options.speaker_reference_path)
            if explicit is None or not explicit.exists():
                raise ValidationError("The provided speaker reference file does not exist.")
            references[None] = explicit

        logger.debug("Resolved %d voice reference(s) for project %s", len(references), project_id)
        return references

    @staticmethod
    def _update_project(project_id: int, **fields: object) -> None:
        if not fields:
            return
        with session_scope() as db:
            project = db.get(DubbingProject, project_id)
            if project is None:
                return
            for key, value in fields.items():
                setattr(project, key, value)


__all__ = ["STEP_MESSAGES", "STEP_TOTAL", "TRANSIENT_STATUSES", "DubbingService"]
