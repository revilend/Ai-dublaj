"""FastAPI application factory and ASGI entrypoint.

Run it with::

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import api_router, ws_router
from app.core.config import Settings, get_settings
from app.core.database import init_db
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger
from app.services import get_storage, get_task_manager
from app.services.storage import MEDIA_URL_PREFIX

logger = get_logger(__name__)

CLEANUP_INTERVAL_SECONDS = 300

_HTTP_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "unprocessable_entity",
    429: "too_many_requests",
    500: "internal_error",
    503: "service_unavailable",
}


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #
async def _task_cleanup_loop() -> None:
    manager = get_task_manager()
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            manager.cleanup()
        except Exception:  # noqa: BLE001 - the janitor must never die loudly
            logger.exception("Task cleanup pass failed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging()
    settings.ensure_directories()
    init_db()

    manager = get_task_manager()
    manager.bind_loop(asyncio.get_running_loop())
    app.state.task_manager = manager
    app.state.settings = settings

    cleanup_task = asyncio.create_task(_task_cleanup_loop(), name="task-cleanup")
    logger.info(
        "%s %s starting (env=%s, data_dir=%s)",
        settings.app_name,
        settings.app_version,
        settings.environment,
        settings.data_dir,
    )
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        logger.info("%s stopped.", settings.app_name)


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
def _jsonable_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only JSON-safe keys from pydantic's error dictionaries."""
    safe: list[dict[str, Any]] = []
    for error in errors:
        item: dict[str, Any] = {
            "type": error.get("type"),
            "loc": list(error.get("loc") or ()),
            "msg": error.get("msg"),
        }
        if "input" in error:
            item["input"] = error["input"]
        safe.append(item)
    return safe


def _error_response(
    status_code: int, code: str, message: str, details: dict[str, Any] | None = None
) -> JSONResponse:
    payload: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        payload["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=payload)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "%s %s -> %s (%s)", request.method, request.url.path, exc.status_code, exc.message
        )
        return _error_response(exc.status_code, exc.code, exc.message, exc.details or None)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            # 422 — spelled numerically because the Starlette constant that
            # holds it was renamed across versions.
            422,
            "request_validation_error",
            "The request payload failed validation.",
            {"errors": _jsonable_errors(exc.errors())},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_error(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "http_error")
        return _error_response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "An internal server error occurred.",
        )


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def _mount_media(app: FastAPI, settings: Settings) -> None:
    """Expose generated media, but never the SQLite file or temp scratch."""
    for name, directory in (
        ("uploads", settings.uploads_dir),
        ("voices", settings.voices_dir),
        ("outputs", settings.outputs_dir),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        app.mount(
            f"{MEDIA_URL_PREFIX}/{name}",
            StaticFiles(directory=str(directory)),
            name=f"media-{name}",
        )


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()
    settings.ensure_directories()

    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        debug=settings.debug and not settings.is_production,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    static_dir = settings.static_dir
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    _mount_media(app, settings)

    templates = Jinja2Templates(directory=str(settings.templates_dir))
    app.state.templates = templates
    app.state.storage = get_storage()

    app.include_router(api_router)
    app.include_router(ws_router)
    register_exception_handlers(app)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def studio(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "app_name": settings.app_name,
                "app_description": settings.app_description,
                "app_version": settings.app_version,
            },
        )

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    logger.info("Application configured: %d routes", len(app.routes))
    return app


app = create_app()


__all__ = ["app", "create_app"]


if __name__ == "__main__":  # pragma: no cover - manual run helper
    import uvicorn

    _settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=_settings.host,
        port=_settings.port,
        reload=_settings.debug and not _settings.is_production,
    )
