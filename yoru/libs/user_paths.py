"""User-directory paths and file logging for YORU.

Centralizes the ``~/.yoru`` user directory used for:

- persistent runtime / error logs (``~/.yoru/logs/yoru.log``), and
- user state such as the last-used condition file
  (``~/.yoru/condition_file_log.json``).

This module intentionally has **no** GUI / OpenCV dependency so it can be
imported early (CLI, app startup) and unit-tested headlessly. The user
directory can be relocated with the ``YORU_HOME`` environment variable
(defaults to ``~/.yoru``).
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_DEFAULT_DIRNAME = ".yoru"
_LOG_FILENAME = "yoru.log"
_STATE_FILENAME = "condition_file_log.json"
_LOGGER_NAME = "yoru"

# Rotating file handler defaults.
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3

_logging_configured = False


def get_yoru_home() -> Path:
    """Return the ``~/.yoru`` directory (overridable via ``YORU_HOME``), creating it."""
    override = os.environ.get("YORU_HOME")
    home = Path(override) if override else Path.home() / _DEFAULT_DIRNAME
    home.mkdir(parents=True, exist_ok=True)
    return home


def get_log_dir() -> Path:
    """Return ``~/.yoru/logs`` (created)."""
    log_dir = get_yoru_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_log_file() -> Path:
    """Return the path to the rotating log file ``~/.yoru/logs/yoru.log``."""
    return get_log_dir() / _LOG_FILENAME


def get_state_file() -> Path:
    """Return the user-state file ``~/.yoru/condition_file_log.json``."""
    return get_yoru_home() / _STATE_FILENAME


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach a rotating file handler writing to ``~/.yoru/logs/yoru.log``.

    Safe to call multiple times; the file handler is attached only once. If the
    log file cannot be opened, logging falls back to console only rather than
    crashing the application.
    """
    global _logging_configured
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    if _logging_configured:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )
    try:
        handler = RotatingFileHandler(
            get_log_file(),
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except Exception:
        # If the log file can't be created, keep going without file logging.
        pass

    _logging_configured = True
    return logger


def get_logger() -> logging.Logger:
    """Return the shared ``yoru`` logger (configuring logging on first use)."""
    setup_logging()
    return logging.getLogger(_LOGGER_NAME)


def log_exception(context: str, exc: BaseException) -> None:
    """Record an exception (with traceback) to the YORU log file."""
    try:
        get_logger().error("%s: %s", context, exc, exc_info=exc)
    except Exception:
        # Logging must never raise into the caller's error-handling path.
        pass


def log_message(message: str, level: int = logging.INFO) -> None:
    """Record a plain message to the YORU log file."""
    try:
        get_logger().log(level, message)
    except Exception:
        pass
