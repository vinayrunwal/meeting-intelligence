"""
Meeting Intelligence System — Structured Logging Configuration
===============================================================
Provides JSON-structured logging for production and pretty console
logging for development. Separate loggers for pipeline, API, and metrics.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

import structlog

from config.settings import settings


def _setup_stdlib_logging(log_dir: Path, level: str) -> None:
    """Configure the standard library root logger with file + stream handlers."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ── Console handler ────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)-24s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(console_fmt)
    root.addHandler(console)

    # ── Rotating file handler ──────────────────────────────────────────
    log_file = log_dir / "meeting_intelligence.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    root.addHandler(file_handler)


def _setup_structlog(json_format: bool = True) -> None:
    """Configure structlog processors for structured logging."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            pad_event=40,
        )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def configure_logging(
    level: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    """
    Initialize the logging system. Call once at application startup.

    Parameters
    ----------
    level : str, optional
        Override log level (DEBUG, INFO, WARNING, ERROR). Defaults to config.
    log_format : str, optional
        Override format ('json' or 'console'). Defaults to config.
    """
    _level = level or settings.log.level
    _format = log_format or settings.log.format

    _setup_stdlib_logging(settings.log.log_dir, _level)
    _setup_structlog(json_format=(_format == "json"))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger bound to the given name.

    Usage::

        from config.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Processing audio", file="meeting.wav", duration=120.5)
    """
    return structlog.get_logger(name)


# Pre-defined logger names for consistency
LOGGER_PIPELINE = "pipeline"
LOGGER_ASR = "pipeline.asr"
LOGGER_DIARIZATION = "pipeline.diarization"
LOGGER_EMOTION = "pipeline.emotion"
LOGGER_INTENT = "pipeline.intent"
LOGGER_DEBRIEF = "pipeline.debrief"
LOGGER_API = "api"
LOGGER_METRICS = "metrics"
