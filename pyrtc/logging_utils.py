from __future__ import annotations

"""Shared logging configuration helpers for pyRTC.

This module defines the common logging surface used by library code, scripts,
benchmarks, and hard-RTC child processes. It centralizes environment-variable
handling, console/file handler configuration, and small CLI helpers so every
entry point can opt into the same operational logging model.
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path


PYRTC_LOG_LEVEL_ENV = "PYRTC_LOG_LEVEL"
PYRTC_LOG_DIR_ENV = "PYRTC_LOG_DIR"
PYRTC_LOG_FILE_ENV = "PYRTC_LOG_FILE"
PYRTC_LOG_COLOR_ENV = "PYRTC_LOG_COLOR"
PYRTC_LOG_CONSOLE_ENV = "PYRTC_LOG_CONSOLE"
PYRTC_LOGGER_NAME = "pyRTC"

_LEVEL_NAMES = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[35m",
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Formatter that optionally colorizes the rendered log level.

    Console and file handlers in pyRTC share the same structured message format.
    This formatter only changes the log-level field when color is enabled, which
    keeps terminal output easier to scan without affecting file logs.
    """
    def __init__(self, use_color: bool):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(processName)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        if self.use_color:
            color = _LEVEL_COLORS.get(record.levelno, "")
            if color:
                record.levelname = f"{color}{original_levelname}{_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def _parse_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_level(level, default: str = "INFO") -> int:
    if level is None:
        level = os.environ.get(PYRTC_LOG_LEVEL_ENV, default)
    if isinstance(level, int):
        return level
    normalized = str(level).strip().upper()
    if normalized.isdigit():
        return int(normalized)
    if normalized not in _LEVEL_NAMES:
        raise ValueError(f"Invalid log level: {level}")
    return _LEVEL_NAMES[normalized]


def _should_use_color(color) -> bool:
    if color is None:
        color = os.environ.get(PYRTC_LOG_COLOR_ENV)
    return _parse_bool(color, default=sys.stderr.isatty())


def _should_log_to_console(console) -> bool:
    if console is None:
        console = os.environ.get(PYRTC_LOG_CONSOLE_ENV)
    return _parse_bool(console, default=True)


def _sanitize_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("._") or "pyrtc"


def _resolve_log_paths(app_name: str, component_name=None, log_dir=None, log_file=None):
    resolved_log_dir = log_dir if log_dir is not None else os.environ.get(PYRTC_LOG_DIR_ENV)
    resolved_log_file = log_file if log_file is not None else os.environ.get(PYRTC_LOG_FILE_ENV)

    if resolved_log_file:
        path = Path(resolved_log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return None, path

    if not resolved_log_dir:
        return None, None

    folder = Path(resolved_log_dir).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    stem_parts = [_sanitize_name(app_name)]
    if component_name:
        stem_parts.append(_sanitize_name(component_name))
    stem_parts.append(str(os.getpid()))
    return folder, folder / ("_".join(stem_parts) + ".log")


def get_logger(name: str | None = None) -> logging.Logger:
    if not name:
        return logging.getLogger(PYRTC_LOGGER_NAME)
    if name == PYRTC_LOGGER_NAME or name.startswith(PYRTC_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PYRTC_LOGGER_NAME}.{name}")


def configure_logging(
    *,
    app_name: str = "pyrtc",
    component_name=None,
    level=None,
    log_dir=None,
    log_file=None,
    color=None,
    console=None,
    export_env: bool = True,
) -> logging.Logger:
    logger = get_logger()
    logger.setLevel(_normalize_level(level))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    if _should_log_to_console(console):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logger.level)
        stream_handler.setFormatter(_ColorFormatter(use_color=_should_use_color(color)))
        logger.addHandler(stream_handler)

    _, resolved_log_file = _resolve_log_paths(app_name, component_name, log_dir, log_file)
    if resolved_log_file is not None:
        file_handler = logging.FileHandler(resolved_log_file, encoding="utf-8")
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(_ColorFormatter(use_color=False))
        logger.addHandler(file_handler)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    if export_env:
        os.environ[PYRTC_LOG_LEVEL_ENV] = logging.getLevelName(logger.level)
        resolved_log_dir = log_dir if log_dir is not None else os.environ.get(PYRTC_LOG_DIR_ENV)
        resolved_log_file_env = log_file if log_file is not None else os.environ.get(PYRTC_LOG_FILE_ENV)
        if resolved_log_dir:
            os.environ[PYRTC_LOG_DIR_ENV] = str(Path(resolved_log_dir).expanduser())
        elif PYRTC_LOG_DIR_ENV in os.environ and resolved_log_file_env:
            os.environ.pop(PYRTC_LOG_DIR_ENV, None)
        if resolved_log_file_env:
            os.environ[PYRTC_LOG_FILE_ENV] = str(Path(resolved_log_file_env).expanduser())
        elif PYRTC_LOG_FILE_ENV in os.environ and log_file is None:
            os.environ.pop(PYRTC_LOG_FILE_ENV, None)
        os.environ[PYRTC_LOG_COLOR_ENV] = "1" if _should_use_color(color) else "0"
        os.environ[PYRTC_LOG_CONSOLE_ENV] = "1" if _should_log_to_console(console) else "0"

    logger.debug(
        "Configured logging app_name=%s component_name=%s level=%s file=%s",
        app_name,
        component_name,
        logging.getLevelName(logger.level),
        resolved_log_file,
    )
    return logger


def ensure_logging_configured(*, app_name: str = "pyrtc", component_name=None) -> logging.Logger:
    logger = get_logger()
    active_handlers = [handler for handler in logger.handlers if not isinstance(handler, logging.NullHandler)]
    if active_handlers:
        return logger
    return configure_logging(app_name=app_name, component_name=component_name, export_env=False)


def add_logging_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--log-level",
        default=None,
        help="Log level override (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to PYRTC_LOG_LEVEL or INFO.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional directory for per-process log files. Defaults to PYRTC_LOG_DIR when set.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional exact log file path. Use with care in multi-process runs.",
    )
    return parser


def configure_logging_from_args(args, *, app_name: str = "pyrtc", component_name=None) -> logging.Logger:
    return configure_logging(
        app_name=app_name,
        component_name=component_name,
        level=getattr(args, "log_level", None),
        log_dir=getattr(args, "log_dir", None),
        log_file=getattr(args, "log_file", None),
    )


get_logger().addHandler(logging.NullHandler())