"""Aggregate API router.

``api_router`` is mounted under ``/api`` by ``app.main``; ``ws_router`` is
mounted at the application root so WebSocket clients use ``/ws/tasks/{id}``.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api import dubbing, synthesis, system, tasks, voices
from app.api.deps import (
    AudioDep,
    DbSession,
    DubbingDep,
    Pagination,
    PaginationDep,
    SettingsDep,
    StorageDep,
    TTSDep,
    TasksDep,
    TranscriberDep,
    TranslatorDep,
)

api_router = APIRouter(prefix="/api")
api_router.include_router(system.router)
api_router.include_router(voices.router)
api_router.include_router(synthesis.router)
api_router.include_router(dubbing.router)
api_router.include_router(tasks.router)

ws_router = APIRouter()
ws_router.include_router(tasks.ws_router)

__all__ = [
    "AudioDep",
    "DbSession",
    "DubbingDep",
    "Pagination",
    "PaginationDep",
    "SettingsDep",
    "StorageDep",
    "TTSDep",
    "TasksDep",
    "TranscriberDep",
    "TranslatorDep",
    "api_router",
    "ws_router",
]
