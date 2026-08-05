"""Logging configuration for the expenses application."""
from __future__ import annotations

import logging

from src.support.paths import LOG_FILE

_LOGGING_CONFIGURED = False


def configure_logging() -> logging.Logger:
    """Configure root logging once and return the application logger."""
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED:
        return logging.getLogger("expenses")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    _LOGGING_CONFIGURED = True
    return logging.getLogger("expenses")
