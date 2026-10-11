"""Voice profile endpoints — create, list, read, update, delete."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Query, UploadFile, status
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import func, select

from app.api.deps import AudioDep, DbSession, PaginationDep, StorageDep
from app.core.config import ALLOWED_AUDIO_EXTENSIONS
from app.core.exceptions import (
    ConflictError,
    MediaProcessingError,
    NotFoundError,
    ValidationError,
)
from app.core.logging import get_logger
from app.models.voice_profile import VoiceProfile
from app.schemas.common import Paginated
from app.schemas.voice import (
    VoiceProfileCreate,
    VoiceProfileRead,
    VoiceProfileUpdate,
)

router = APIRouter(prefix="/voices", tags=["voices"])
logger = get_logger(__name__)


def _serialize(voice: VoiceProfile, storage: StorageDep) -> VoiceProfileRead:
    return VoiceProfileRead.from_model(
        voice, sample_audio_url=storage.url_for(voice.sample_audio_path)
    )


@router.post(
    "",
    response_model=VoiceProfileRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a reusable voice profile",
    description=(
        "Upload a short (6–30 s) clean reference clip. The clip is stored and "
        "reused by XTTS-v2 every time this voice is asked to speak."
    ),
)
async def create_voice(
    name: Annotated[str, Form(min_length=1, max_length=150)],
    sample_audio: Annotated[UploadFile, File(description="Reference audio (wav/mp3/flac/…)")],
    db: DbSession,
    storage: StorageDep,
    audio: AudioDep,
    description: Annotated[str | None, Form(max_length=2000)] = None,
    language: Annotated[str, Form(min_length=2, max_length=16)] = "en",
) -> VoiceProfileRead:
    try:
        payload = VoiceProfileCreate(
            name=name, description=description, language=language.strip().lower()
        )
    except PydanticValidationError as exc:
        raise ValidationError(
            "Invalid voice profile payload.",
            details={"errors": exc.errors(include_url=False)},
        ) from exc

    existing = db.scalar(select(VoiceProfile).where(VoiceProfile.name == payload.name))
    if existing is not None:
        raise ConflictError(
            f"A voice profile named '{payload.name}' already exists.",
            details={"voice_profile_id": existing.id},
        )

    stored, _size = await storage.save_upload(
        sample_audio,
        storage.voices_dir,
        allowed_extensions=ALLOWED_AUDIO_EXTENSIONS,
        prefix=f"{payload.name[:40]}-",
    )

    duration: float | None = None
    if audio.ffprobe_available:
        try:
            info = audio.probe(stored)
            if not info.has_audio:
                storage.delete(stored)
                raise ValidationError(
                    "The uploaded reference file does not contain an audio stream."
                )
            duration = info.duration
        except MediaProcessingError as exc:
            storage.delete(stored)
            raise exc

    voice = VoiceProfile(
        name=payload.name,
        description=payload.description,
        sample_audio_path=storage.relative(stored),
        language=payload.language,
        duration_seconds=duration,
    )
    db.add(voice)
    db.commit()
    db.refresh(voice)
    logger.info("Created voice profile %s (%s)", voice.id, voice.name)
    return _serialize(voice, storage)


@router.get(
    "",
    response_model=Paginated[VoiceProfileRead],
    summary="List saved voice profiles",
)
def list_voices(
    db: DbSession,
    storage: StorageDep,
    pagination: PaginationDep,
    search: Annotated[str | None, Query(max_length=150)] = None,
    active_only: Annotated[bool, Query()] = False,
) -> Paginated[VoiceProfileRead]:
    query = select(VoiceProfile)
    if search:
        query = query.where(VoiceProfile.name.ilike(f"%{search.strip()}%"))
    if active_only:
        query = query.where(VoiceProfile.is_active.is_(True))

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = db.scalars(
        query.order_by(VoiceProfile.created_at.desc())
        .offset(pagination.skip)
        .limit(pagination.limit)
    ).all()

    return Paginated[VoiceProfileRead](
        items=[_serialize(row, storage) for row in rows],
        total=int(total),
        skip=pagination.skip,
        limit=pagination.limit,
    )


@router.get("/{voice_id}", response_model=VoiceProfileRead, summary="Get a voice profile")
def get_voice(voice_id: int, db: DbSession, storage: StorageDep) -> VoiceProfileRead:
    voice = db.get(VoiceProfile, voice_id)
    if voice is None:
        raise NotFoundError(
            f"Voice profile {voice_id} does not exist.", details={"voice_id": voice_id}
        )
    return _serialize(voice, storage)


@router.patch("/{voice_id}", response_model=VoiceProfileRead, summary="Update a voice profile")
def update_voice(
    voice_id: int,
    payload: VoiceProfileUpdate,
    db: DbSession,
    storage: StorageDep,
) -> VoiceProfileRead:
    voice = db.get(VoiceProfile, voice_id)
    if voice is None:
        raise NotFoundError(
            f"Voice profile {voice_id} does not exist.", details={"voice_id": voice_id}
        )

    updates = payload.model_dump(exclude_unset=True, exclude_none=True)
    if "name" in updates and updates["name"] != voice.name:
        clash = db.scalar(select(VoiceProfile).where(VoiceProfile.name == updates["name"]))
        if clash is not None:
            raise ConflictError(f"A voice profile named '{updates['name']}' already exists.")
    if "language" in updates:
        updates["language"] = str(updates["language"]).strip().lower()

    for key, value in updates.items():
        setattr(voice, key, value)

    db.commit()
    db.refresh(voice)
    return _serialize(voice, storage)


@router.delete(
    "/{voice_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a voice profile",
)
def delete_voice(voice_id: int, db: DbSession, storage: StorageDep) -> None:
    voice = db.get(VoiceProfile, voice_id)
    if voice is None:
        raise NotFoundError(
            f"Voice profile {voice_id} does not exist.", details={"voice_id": voice_id}
        )
    sample_path = voice.sample_audio_path
    db.delete(voice)
    db.commit()
    storage.delete(sample_path)
    logger.info("Deleted voice profile %s", voice_id)


__all__ = ["router"]
