"""File storage helpers: safe uploads, deterministic paths, public URLs."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.core.exceptions import StorageError, ValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_CHUNK_SIZE = 1024 * 1024  # 1 MiB
MEDIA_URL_PREFIX = "/media"


def sanitize_filename(filename: str | None, *, fallback: str = "upload") -> str:
    """Return a filesystem-safe basename, never empty and never a path."""
    candidate = Path(filename or "").name
    candidate = _SAFE_NAME.sub("_", candidate).strip("._")
    if not candidate:
        candidate = fallback
    return candidate[:150]


class StorageManager:
    """Owns everything written under the configured data directory."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.root = self.settings.data_dir.resolve()

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    @property
    def uploads_dir(self) -> Path:
        return self.settings.uploads_dir

    @property
    def voices_dir(self) -> Path:
        return self.settings.voices_dir

    @property
    def outputs_dir(self) -> Path:
        return self.settings.outputs_dir

    @property
    def temp_dir(self) -> Path:
        return self.settings.temp_dir

    def ensure_directories(self) -> None:
        self.settings.ensure_directories()

    def unique_path(self, directory: Path, *, suffix: str, prefix: str = "") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        stem = f"{prefix}{uuid.uuid4().hex[:12]}"
        return directory / f"{stem}{suffix}"

    def stage_dir(self, name: str) -> Path:
        """Create (or reuse) a scratch directory for one job."""
        directory = self.temp_dir / f"{sanitize_filename(name, fallback='job')}-{uuid.uuid4().hex[:8]}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def relative(self, path: str | Path) -> str:
        """Store paths relative to the data root so the DB stays portable."""
        resolved = Path(path).resolve()
        for base in (self.root, PROJECT_ROOT.resolve()):
            try:
                return resolved.relative_to(base).as_posix()
            except ValueError:
                continue
        return resolved.as_posix()

    def absolute(self, stored: str | Path | None) -> Path | None:
        """Resolve a stored (relative) path back to an absolute one."""
        if stored in (None, ""):
            return None
        candidate = Path(str(stored))
        if candidate.is_absolute():
            return candidate
        for base in (self.root, PROJECT_ROOT.resolve()):
            resolved = (base / candidate).resolve()
            if resolved.exists():
                return resolved
        return (self.root / candidate).resolve()

    def url_for(self, stored: str | Path | None) -> str | None:
        """Public URL for a stored path (served by the ``/media`` mount)."""
        if stored in (None, ""):
            return None
        resolved = self.absolute(stored)
        if resolved is None:
            return None
        try:
            relative = resolved.relative_to(self.root).as_posix()
        except ValueError:
            return None
        return f"{MEDIA_URL_PREFIX}/{relative}"

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    async def save_upload(
        self,
        upload: UploadFile,
        directory: Path,
        *,
        allowed_extensions: frozenset[str],
        prefix: str = "",
        max_bytes: int | None = None,
    ) -> tuple[Path, int]:
        """Stream an ``UploadFile`` to disk, enforcing size and extension."""
        filename = sanitize_filename(upload.filename, fallback="upload")
        suffix = Path(filename).suffix.lower()
        if suffix not in allowed_extensions:
            raise ValidationError(
                f"Unsupported file type '{suffix or filename}'.",
                details={"allowed_extensions": sorted(allowed_extensions)},
            )

        limit = max_bytes or self.settings.max_upload_size_bytes
        directory.mkdir(parents=True, exist_ok=True)
        destination = self.unique_path(directory, suffix=suffix, prefix=prefix or f"{Path(filename).stem[:40]}-")

        written = 0
        try:
            await upload.seek(0)
            async with aiofiles.open(destination, "wb") as handle:
                while True:
                    chunk = await upload.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > limit:
                        raise ValidationError(
                            "Uploaded file exceeds the maximum allowed size.",
                            details={"max_upload_size_mb": self.settings.max_upload_size_mb},
                        )
                    await handle.write(chunk)
        except ValidationError:
            destination.unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001 - normalise I/O failures
            destination.unlink(missing_ok=True)
            logger.exception("Failed to store upload %s", filename)
            raise StorageError(f"Could not store '{filename}': {exc}") from exc
        finally:
            await upload.close()

        if written == 0:
            destination.unlink(missing_ok=True)
            raise ValidationError("Uploaded file is empty.")

        logger.info("Stored upload %s (%d bytes)", destination.name, written)
        return destination, written

    def save_bytes(self, payload: bytes, directory: Path, *, suffix: str, prefix: str = "") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        destination = self.unique_path(directory, suffix=suffix, prefix=prefix)
        destination.write_bytes(payload)
        return destination

    def save_text(self, payload: str, directory: Path, *, suffix: str, prefix: str = "") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        destination = self.unique_path(directory, suffix=suffix, prefix=prefix)
        destination.write_text(payload, encoding="utf-8")
        return destination

    def delete(self, stored: str | Path | None) -> bool:
        """Best-effort deletion of a stored file (never raises)."""
        path = self.absolute(stored)
        if path is None or not path.exists():
            return False
        try:
            if path.is_dir():
                for child in sorted(path.rglob("*"), reverse=True):
                    child.unlink(missing_ok=True) if child.is_file() else child.rmdir()
                path.rmdir()
            else:
                path.unlink()
            return True
        except OSError:  # pragma: no cover - defensive
            logger.warning("Could not delete %s", path)
            return False


__all__ = ["MEDIA_URL_PREFIX", "StorageManager", "sanitize_filename"]
