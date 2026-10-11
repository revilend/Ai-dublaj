"""Structured logging setup shared by the API and the service layer."""

from __future__ import annotations

import logging
import sys

from app.core.config import get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once; safe to call multiple times."""
    global _configured

    settings = get_settings()
    resolved = (level or settings.log_level).upper()
    numeric_level = getattr(logging, resolved, logging.INFO)

    root = logging.getLogger()
    if not _configured:
        # Respect handlers installed by uvicorn/gunicorn if they already exist.
        if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
            root.addHandler(handler)
        _configured = True

    root.setLevel(numeric_level)
    logging.getLogger("app").setLevel(numeric_level)

    # Keep third-party libraries from drowning out our own logs.
    for noisy in ("httpx", "urllib3", "asyncio", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
