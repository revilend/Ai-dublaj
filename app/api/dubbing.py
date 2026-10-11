"""Dubbing Studio endpoints.

The public surface follows the studio workflow:

===========================  ==================================================
``POST /api/dubbing/upload``  Upload a video and (optionally) run the analysis
                              phase: extraction → separation → transcription.
``PUT  /api/dubbing/segments/{segment_id}``
                              Edit a cue's text/timing or assign it a different
                              saved voice profile.
``POST /api/dubbing/render/{project_id}``
                              Synthesise voices, time-stretch to fit, mix with
                              ``background.wav`` and export the final video.
===========================  ==================================================

Plus the supporting reads: list/get/delete a project, list and bulk-add
segments, import an SRT/VTT, and delete one segment.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Query, UploadFile, status
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import func, select

from app.api.deps import DbSession, DubbingDep, PaginationDep, StorageDep, TasksDep
from app.core.config import ALLOWED_SUBTITLE_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS
from app.core.exceptions import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.models.dubbing_project import DubbingProject, DubbingStatus
from app.models.subtitle_segment import SubtitleSegment
from app.models.voice_profile import VoiceProfile
from app.schemas.common import Paginated, TaskAccepted
from app.schemas.dubbing import (
    DubbingProjectCreate,
    DubbingProjectDetail,
    DubbingProjectRead,
    DubbingUploadResponse,
    RenderRequest,
    SegmentBulkCreate,
    SubtitleSegmentCreate,
    SubtitleSegmentRead,
    SubtitleSegmentUpdate,
)
from app.services import StorageManager, parse_subtitles, to_srt

router = APIRouter(prefix="/dubbing", tags=["dubbing"])
logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #
def _serialize_project(project: DubbingProject, storage: StorageManager) -> DubbingProjectRead:
    payload = DubbingProjectRead.model_validate(project)
    payload.video_url = storage.url_for(project.video_path)
    payload.final_video_url = storage.url_for(project.final_video_path)
    payload.vocals_audio_url = storage.url_for(project.vocals_audio_path)
    payload.background_audio_url = storage.url_for(project.background_audio_path)
    payload.dubbed_audio_url = storage.url_for(project.dubbed_audio_path)
    return payload


def _serialize_segment(segment: SubtitleSegment, storage: StorageManager) -> SubtitleSegmentRead:
    payload = SubtitleSegmentRead.model_validate(segment)
    payload.dubbed_audio_url = storage.url_for(segment.dubbed_audio_path)
    payload.voice_profile_name = segment.voice_profile.name if segment.voice_profile else None
    return payload


def _get_project(db: DbSession, project_id: int) -> DubbingProject:
    project = db.get(DubbingProject, project_id)
    if project is None:
        raise NotFoundError(
            f"Dubbing project {project_id} does not exist.",
            details={"project_id": project_id},
        )
    return project


def _list_segments(db: DbSession, project_id: int) -> list[SubtitleSegment]:
    return list(
        db.scalars(
            select(SubtitleSegment)
            .where(SubtitleSegment.project_id == project_id)
            .order_by(SubtitleSegment.start_time)
        )
    )


def _require_voice_profile(db: DbSession, voice_profile_id: int) -> VoiceProfile:
    profile = db.get(VoiceProfile, voice_profile_id)
    if profile is None:
        raise NotFoundError(
            f"Voice profile {voice_profile_id} does not exist.",
            details={"voice_profile_id": voice_profile_id},
        )
    return profile


# --------------------------------------------------------------------------- #
# Upload + analyse
# --------------------------------------------------------------------------- #
@router.post(
    "/upload",
    response_model=DubbingUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a video and start the analysis phase",
    description=(
        "Saves the video, then queues **Step 1/4: Separating Music** (FFmpeg "
        "extraction + Demucs → `vocals.wav` / `background.wav`) and "
        "**Step 2/4: Transcribing** (faster-whisper → timed segments).\n\n"
        "The project id is returned immediately; follow `status_url` or "
        "`websocket_url` for progress, then read the parsed segments from "
        "`GET /api/dubbing/{project_id}`.\n\n"
        "Pass `analyze=false` to create the project without running the pipeline."
    ),
)
async def upload_video(
    title: Annotated[str, Form(min_length=1, max_length=200)],
    video: Annotated[UploadFile, File(description="Source video (mp4/mov/mkv/webm/…)")],
    db: DbSession,
    storage: StorageDep,
    tasks: TasksDep,
    dubbing: DubbingDep,
    source_language: Annotated[str, Form(min_length=2, max_length=16)] = "auto",
    target_language: Annotated[str, Form(min_length=2, max_length=16)] = "en",
    voice_profile_id: Annotated[int | None, Form(gt=0)] = None,
    analyze: Annotated[bool, Form()] = True,
) -> DubbingUploadResponse:
    try:
        payload = DubbingProjectCreate(
            title=title,
            source_language=source_language.strip().lower(),
            target_language=target_language.strip().lower(),
            voice_profile_id=voice_profile_id,
        )
    except PydanticValidationError as exc:
        raise ValidationError(
            "Invalid project payload.",
            details={"errors": exc.errors(include_url=False)},
        ) from exc

    if payload.voice_profile_id is not None:
        _require_voice_profile(db, payload.voice_profile_id)

    stored, _size = await storage.save_upload(
        video,
        storage.uploads_dir,
        allowed_extensions=ALLOWED_VIDEO_EXTENSIONS,
        prefix="video-",
    )

    project = DubbingProject(
        title=payload.title,
        video_path=storage.relative(stored),
        source_language=payload.source_language,
        target_language=payload.target_language,
        voice_profile_id=payload.voice_profile_id,
        status=DubbingStatus.PENDING,
        progress=0.0,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    logger.info("Created dubbing project %s (%s)", project.id, project.title)

    if not analyze:
        return DubbingUploadResponse(
            project_id=project.id,
            title=project.title,
            status=project.status,
            message="Project created. Run the analysis phase when ready.",
        )

    record = tasks.create(
        f"Analyze · {project.title}",
        kind="dubbing-analyze",
        metadata={"project_id": project.id, "step_total": 4, "step": 1},
    )
    project.last_task_id = record.id
    db.commit()

    async def body(ctx) -> dict[str, object]:
        return await dubbing.analyze(ctx, project.id)

    tasks.start(record, body)

    return DubbingUploadResponse(
        project_id=project.id,
        title=project.title,
        status=project.status,
        task_id=record.id,
        status_url=f"/api/tasks/{record.id}/status",
        websocket_url=f"/ws/tasks/{record.id}",
        message="Upload accepted. Extracting audio and transcribing.",
    )


# --------------------------------------------------------------------------- #
# Segment editing
# --------------------------------------------------------------------------- #
@router.put(
    "/segments/{segment_id}",
    response_model=SubtitleSegmentRead,
    summary="Update a segment's text, timing or voice",
    description=(
        "Partial update. Send `translated_text` to correct a translation, or "
        "`voice_profile_id` to give this one segment a different saved voice "
        "(send `null` to fall back to the project voice).\n\n"
        "Changing the text or the voice invalidates the segment's previously "
        "generated audio, which is deleted so a re-render regenerates it."
    ),
)
def update_segment(
    segment_id: int,
    payload: SubtitleSegmentUpdate,
    db: DbSession,
    storage: StorageDep,
) -> SubtitleSegmentRead:
    segment = db.get(SubtitleSegment, segment_id)
    if segment is None:
        raise NotFoundError(
            f"Subtitle segment {segment_id} does not exist.",
            details={"segment_id": segment_id},
        )

    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        return _serialize_segment(segment, storage)

    if "voice_profile_id" in updates and updates["voice_profile_id"] is not None:
        _require_voice_profile(db, int(updates["voice_profile_id"]))

    start = float(updates.get("start_time", segment.start_time))
    end = float(updates.get("end_time", segment.end_time))
    if end < start:
        raise ValidationError(
            "end_time must be greater than or equal to start_time.",
            details={"start_time": start, "end_time": end},
        )

    # Regenerating audio is the only way to reflect new text/voice, so drop the
    # stale artifact rather than leaving it to be silently reused.
    invalidating = {"original_text", "translated_text", "voice_profile_id", "start_time", "end_time"}
    needs_regen = bool(invalidating & updates.keys()) and bool(segment.dubbed_audio_path)

    for key, value in updates.items():
        setattr(segment, key, value)

    stale_path = segment.dubbed_audio_path if needs_regen else None
    if needs_regen:
        segment.dubbed_audio_path = None

    db.commit()
    db.refresh(segment)
    if stale_path:
        storage.delete(stale_path)
        logger.info("Segment %s updated; stale dubbed audio removed", segment_id)

    return _serialize_segment(segment, storage)


# --------------------------------------------------------------------------- #
# Per-line preview
# --------------------------------------------------------------------------- #
@router.post(
    "/segments/{segment_id}/preview",
    response_model=TaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Synthesise a single line for auditioning",
    description=(
        "Clones **only** the given cue with the voice profile it resolves to "
        "(segment voice → project voice) and stores the clip as that segment's "
        "audio. This is the editor's *Preview line* action: it lets a translation "
        "be heard and corrected without rendering the whole video.\n\n"
        "Follow `status_url` / `websocket_url`, then re-read the segment — "
        "`dubbed_audio_url` becomes playable once the task completes."
    ),
)
async def preview_segment(
    segment_id: int,
    db: DbSession,
    tasks: TasksDep,
    dubbing: DubbingDep,
    options: RenderRequest | None = None,
) -> TaskAccepted:
    segment = db.get(SubtitleSegment, segment_id)
    if segment is None:
        raise NotFoundError(
            f"Subtitle segment {segment_id} does not exist.",
            details={"segment_id": segment_id},
        )
    options = options or RenderRequest()
    if options.voice_profile_id is not None:
        _require_voice_profile(db, options.voice_profile_id)

    project = _get_project(db, segment.project_id)
    if (
        segment.voice_profile_id is None
        and project.voice_profile_id is None
        and options.voice_profile_id is None
        and not options.speaker_reference_path
    ):
        raise ValidationError(
            "Assign a voice profile to this line (or set a project voice) before previewing it.",
            details={"segment_id": segment_id, "hint": "PUT /api/dubbing/segments/{id}"},
        )

    record = tasks.create(
        f"Preview line {segment.order_index + 1}",
        kind="dubbing-preview",
        metadata={
            "project_id": project.id,
            "segment_id": segment_id,
            "step_total": 4,
            "step": 3,
        },
    )

    async def body(ctx) -> dict[str, object]:
        return await dubbing.preview_segment(ctx, segment_id, options)

    tasks.start(record, body)

    return TaskAccepted(
        task_id=record.id,
        status=record.status.value,
        progress=record.progress,
        message="Previewing the line.",
        status_url=f"/api/tasks/{record.id}/status",
        websocket_url=f"/ws/tasks/{record.id}",
        resource_id=segment_id,
    )


# --------------------------------------------------------------------------- #
# Auto-translate (editor step, no synthesis)
# --------------------------------------------------------------------------- #
@router.post(
    "/{project_id}/translate",
    response_model=TaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Auto-translate every segment",
    description=(
        "Fills `translated_text` for the whole subtitle track with NLLB and "
        "returns immediately (`202`). Nothing is synthesised, and the project is "
        "left ready to render — so translations can be hand-edited first."
    ),
)
async def translate_project(
    project_id: int,
    db: DbSession,
    tasks: TasksDep,
    dubbing: DubbingDep,
    options: RenderRequest | None = None,
) -> TaskAccepted:
    project = _get_project(db, project_id)
    options = options or RenderRequest()

    record = tasks.create(
        f"Translate · {project.title}",
        kind="dubbing-translate",
        metadata={"project_id": project_id, "step_total": 4, "step": 3},
    )
    project.last_task_id = record.id
    db.commit()

    async def body(ctx) -> dict[str, object]:
        return await dubbing.translate_segments(ctx, project_id, options)

    tasks.start(record, body)

    return TaskAccepted(
        task_id=record.id,
        status=record.status.value,
        progress=record.progress,
        message="Translating the subtitle track.",
        status_url=f"/api/tasks/{record.id}/status",
        websocket_url=f"/ws/tasks/{record.id}",
        resource_id=project_id,
    )


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
@router.post(
    "/render/{project_id}",
    response_model=TaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Synthesise and export the dubbed video",
    description=(
        "Queues **Step 3/4: Generating Voices** (XTTS-v2 per segment, "
        "time-stretched to fit each cue) and **Step 4/4: Rendering Video** "
        "(rebuild the timeline, mix `background.wav` back in, mux with FFmpeg).\n\n"
        "Steps 1–2 are re-run automatically if their artifacts are missing, so "
        "this endpoint is safe to call on a freshly uploaded project."
    ),
)
async def render_project(
    project_id: int,
    db: DbSession,
    storage: StorageDep,
    tasks: TasksDep,
    dubbing: DubbingDep,
    options: RenderRequest | None = None,
) -> TaskAccepted:
    project = _get_project(db, project_id)
    options = options or RenderRequest()

    if options.voice_profile_id is not None:
        _require_voice_profile(db, options.voice_profile_id)

    has_segment_voice = db.scalar(
        select(func.count())
        .select_from(SubtitleSegment)
        .where(
            SubtitleSegment.project_id == project_id,
            SubtitleSegment.voice_profile_id.is_not(None),
        )
    )
    if (
        project.voice_profile_id is None
        and options.voice_profile_id is None
        and not project.speaker_reference_path
        and not options.speaker_reference_path
        and not has_segment_voice
    ):
        raise ValidationError(
            "Select a voice profile (or upload a reference clip) before rendering.",
            details={"hint": "POST /api/voices, then pass voice_profile_id."},
        )

    record = tasks.create(
        f"Dub · {project.title}",
        kind="dubbing-render",
        metadata={"project_id": project_id, "step_total": 4, "step": 3},
    )
    project.last_task_id = record.id
    db.commit()

    async def body(ctx) -> dict[str, object]:
        return await dubbing.render(ctx, project_id, options)

    tasks.start(record, body)

    return TaskAccepted(
        task_id=record.id,
        status=record.status.value,
        progress=record.progress,
        message="Dubbing queued.",
        status_url=f"/api/tasks/{record.id}/status",
        websocket_url=f"/ws/tasks/{record.id}",
        resource_id=project_id,
    )


# --------------------------------------------------------------------------- #
# Project reads
# --------------------------------------------------------------------------- #
@router.get("", response_model=Paginated[DubbingProjectRead], summary="List dubbing projects")
def list_projects(
    db: DbSession,
    storage: StorageDep,
    pagination: PaginationDep,
    project_status: Annotated[DubbingStatus | None, Query(alias="status")] = None,
) -> Paginated[DubbingProjectRead]:
    query = select(DubbingProject)
    if project_status is not None:
        query = query.where(DubbingProject.status == project_status)

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = db.scalars(
        query.order_by(DubbingProject.created_at.desc())
        .offset(pagination.skip)
        .limit(pagination.limit)
    ).all()

    return Paginated[DubbingProjectRead](
        items=[_serialize_project(row, storage) for row in rows],
        total=int(total),
        skip=pagination.skip,
        limit=pagination.limit,
    )


@router.get("/{project_id}", response_model=DubbingProjectDetail, summary="Get a project")
def get_project(project_id: int, db: DbSession, storage: StorageDep) -> DubbingProjectDetail:
    project = _get_project(db, project_id)
    detail = DubbingProjectDetail.model_validate(project)
    detail.video_url = storage.url_for(project.video_path)
    detail.final_video_url = storage.url_for(project.final_video_path)
    detail.vocals_audio_url = storage.url_for(project.vocals_audio_path)
    detail.background_audio_url = storage.url_for(project.background_audio_path)
    detail.dubbed_audio_url = storage.url_for(project.dubbed_audio_path)
    detail.segments = [_serialize_segment(segment, storage) for segment in project.segments]
    return detail


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a project and its artifacts",
)
def delete_project(project_id: int, db: DbSession, storage: StorageDep) -> None:
    project = _get_project(db, project_id)
    artifacts = [
        project.video_path,
        project.original_audio_path,
        project.vocals_audio_path,
        project.background_audio_path,
        project.dubbed_audio_path,
        project.final_video_path,
    ]
    db.delete(project)
    db.commit()
    for artifact in artifacts:
        storage.delete(artifact)
    # Also drop the per-project scratch/output directory (transcripts, stems…).
    storage.delete(storage.outputs_dir / f"project-{project_id}")
    logger.info("Deleted dubbing project %s", project_id)


# --------------------------------------------------------------------------- #
# Segments
# --------------------------------------------------------------------------- #
@router.get(
    "/{project_id}/segments",
    response_model=list[SubtitleSegmentRead],
    summary="List subtitle segments",
)
def list_segments(project_id: int, db: DbSession, storage: StorageDep) -> list[SubtitleSegmentRead]:
    _get_project(db, project_id)
    return [_serialize_segment(row, storage) for row in _list_segments(db, project_id)]


@router.post(
    "/{project_id}/segments",
    response_model=list[SubtitleSegmentRead],
    status_code=status.HTTP_201_CREATED,
    summary="Add subtitle segments",
)
def add_segments(
    project_id: int,
    payload: SegmentBulkCreate | SubtitleSegmentCreate,
    db: DbSession,
    storage: StorageDep,
    replace: Annotated[bool, Query(description="Replace all existing segments.")] = False,
) -> list[SubtitleSegmentRead]:
    _get_project(db, project_id)
    incoming = payload.segments if isinstance(payload, SegmentBulkCreate) else [payload]

    if replace:
        db.query(SubtitleSegment).filter(SubtitleSegment.project_id == project_id).delete()

    for index, cue in enumerate(incoming):
        db.add(
            SubtitleSegment(
                project_id=project_id,
                start_time=cue.start_time,
                end_time=cue.end_time,
                order_index=index,
                original_text=cue.original_text,
                translated_text=cue.translated_text,
                speaker_id=cue.speaker_id,
            )
        )

    total = db.scalar(
        select(func.count())
        .select_from(SubtitleSegment)
        .where(SubtitleSegment.project_id == project_id)
    )
    project = _get_project(db, project_id)
    project.segment_count = int(total or 0)
    db.commit()

    return [_serialize_segment(row, storage) for row in _list_segments(db, project_id)]


@router.post(
    "/{project_id}/subtitles",
    response_model=list[SubtitleSegmentRead],
    status_code=status.HTTP_201_CREATED,
    summary="Upload an SRT/VTT file",
)
async def upload_subtitles(
    project_id: int,
    file: Annotated[UploadFile, File(description="Subtitle file (.srt or .vtt)")],
    db: DbSession,
    storage: StorageDep,
) -> list[SubtitleSegmentRead]:
    _get_project(db, project_id)

    if not file.filename:
        raise ValidationError("A subtitle file is required.")
    if not file.filename.lower().endswith(tuple(ALLOWED_SUBTITLE_EXTENSIONS)):
        raise ValidationError(
            "Only .srt and .vtt subtitle files are supported.",
            details={"allowed_extensions": sorted(ALLOWED_SUBTITLE_EXTENSIONS)},
        )

    raw = await file.read()
    await file.close()
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw.decode("latin-1", errors="replace")

    cues = parse_subtitles(content)
    storage.save_text(
        to_srt(cues),
        storage.outputs_dir / f"project-{project_id}",
        suffix=".srt",
        prefix="subtitles-",
    )

    db.query(SubtitleSegment).filter(SubtitleSegment.project_id == project_id).delete()
    for index, cue in enumerate(cues):
        db.add(
            SubtitleSegment(
                project_id=project_id,
                start_time=cue.start,
                end_time=cue.end,
                order_index=index,
                original_text=cue.text,
            )
        )
    project = _get_project(db, project_id)
    project.segment_count = len(cues)
    db.commit()

    logger.info("Imported %d subtitle cue(s) into project %s", len(cues), project_id)
    return [_serialize_segment(row, storage) for row in _list_segments(db, project_id)]


@router.delete(
    "/{project_id}/segments/{segment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete one segment",
)
def delete_segment(project_id: int, segment_id: int, db: DbSession, storage: StorageDep) -> None:
    segment = db.get(SubtitleSegment, segment_id)
    if segment is None or segment.project_id != project_id:
        raise NotFoundError(
            f"Subtitle segment {segment_id} does not exist in project {project_id}."
        )
    dubbed = segment.dubbed_audio_path
    db.delete(segment)
    project = _get_project(db, project_id)
    project.segment_count = max(0, project.segment_count - 1)
    db.commit()
    storage.delete(dubbed)


__all__ = ["router"]
