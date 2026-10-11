"""SQLAlchemy engine, session factory and FastAPI session dependency."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_settings = get_settings()

_connect_args: dict[str, object] = {}
if _settings.resolved_database_url.startswith("sqlite"):
    # SQLite connections are bound to the creating thread by default; FastAPI
    # runs sync endpoints and background tasks on different threads.
    _connect_args["check_same_thread"] = False

engine: Engine = create_engine(
    _settings.resolved_database_url,
    connect_args=_connect_args,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


@event.listens_for(engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection: object, _record: object) -> None:
    """SQLite does not enforce foreign keys unless asked to."""
    if not _settings.resolved_database_url.startswith("sqlite"):
        return
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


# Columns added to already-released tables. `create_all` only creates missing
# tables, so new columns on existing tables need an explicit ALTER. SQLite
# permits ADD COLUMN with a REFERENCES clause as long as the default is NULL,
# which keeps the foreign key intact instead of degrading to a bare INTEGER.
_COLUMN_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("subtitle_segments", "voice_profile_id", "INTEGER REFERENCES voice_profiles(id) ON DELETE SET NULL"),
    ("dubbing_projects", "vocals_audio_path", "VARCHAR(1024)"),
)


def apply_column_migrations(target_engine: Engine | None = None) -> list[str]:
    """Add columns introduced after a table was first created.

    Returns the list of ``table.column`` pairs that were added. Safe to call on
    every start — existing columns are skipped, and tables that do not exist
    yet are left to ``create_all`` (which builds them complete).
    """
    active_engine = target_engine or engine
    inspector = inspect(active_engine)
    existing_tables = set(inspector.get_table_names())
    applied: list[str] = []

    for table, column, ddl in _COLUMN_MIGRATIONS:
        if table not in existing_tables:
            continue
        if column in {item["name"] for item in inspector.get_columns(table)}:
            continue
        with active_engine.begin() as connection:
            connection.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {ddl}'))
        applied.append(f"{table}.{column}")
        logger.info("Schema migration applied: %s.%s", table, column)

    return applied


def init_db() -> None:
    """Create all tables. Importing ``app.models`` registers every mapper."""
    from app import models  # noqa: F401  (side-effect: mapper registration)
    from app.models.base import Base

    Base.metadata.create_all(bind=engine)
    apply_column_migrations()
    logger.info("Database initialised at %s", _settings.resolved_database_url)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a scoped session."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session context for background tasks and scripts."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


__all__ = [
    "SessionLocal",
    "apply_column_migrations",
    "engine",
    "get_db",
    "init_db",
    "session_scope",
]
