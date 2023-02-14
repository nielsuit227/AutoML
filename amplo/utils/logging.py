#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import colorlog

__all__ = [
    "add_file_handler",
    "del_file_handlers",
    "get_root_logger",
    "del_root_logger",
]


# ------------------------------------------------------------------------------
# Filters


class TimeFilter(logging.Filter):
    def filter(self, record):
        # Check if previous logged
        if not hasattr(self, "last"):
            self.last = record.relativeCreated

        # Calc & add delta
        delta = datetime.fromtimestamp(
            record.relativeCreated / 1000.0
        ) - datetime.fromtimestamp(self.last / 1000.0)
        record.relative = f"{(delta.seconds + delta.microseconds / 1e6):.2f}"

        # Update last
        self.last = record.relativeCreated

        return True


class NameFilter(logging.Filter):
    """
    Logging filter that ignores child names of loggers that inherit from ``AutoML``.
    """

    def filter(self, record):
        split_name = record.name.split(".", 1)
        if split_name[0] == "AmploML":
            record.name = split_name[0]
        return True


def _add_filters(handler: logging.Handler) -> None:
    handler.addFilter(TimeFilter())
    handler.addFilter(NameFilter())


# ------------------------------------------------------------------------------
# Loggers

_ROOT_LOGGER: logging.Logger | None = None


def _create_logger() -> logging.Logger:
    """
    Creates a new logger that also captures warnings from `warnings.warn()`.

    Returns
    -------
    logging.Logger
        New logger instance.
    """

    # Get custom logger
    logger = logging.getLogger("AmploML")
    logger.setLevel("INFO")

    # Set console handler
    console_formatter = colorlog.ColoredFormatter(
        "%(white)s%(asctime)s %(blue)s[%(name)s]%(log_color)s[%(levelname)s] "
        "%(white)s%(message)s %(light_black)s<%(filename)s:%(lineno)d> (%(relative)ss)",
        datefmt="%H:%M",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.NOTSET)
    console_handler.setFormatter(console_formatter)
    _add_filters(console_handler)
    logger.addHandler(console_handler)

    # Capture warnings from `warnings.warn(...)`
    logging.captureWarnings(True)
    py_warnings_logger = logging.getLogger("py.warnings")
    warnings_formatter = colorlog.ColoredFormatter(
        "%(white)s%(asctime)s %(blue)s[%(name)s]%(log_color)s[%(levelname)s] "
        "%(white)s%(message)s",
        datefmt="%H:%M",
    )
    warnings_handler = logging.StreamHandler()
    warnings_handler.setLevel(logging.WARNING)
    warnings_handler.setFormatter(warnings_formatter)
    warnings_handler.terminator = ""  # suppress unnecessary newline
    _add_filters(warnings_handler)
    py_warnings_logger.addHandler(warnings_handler)

    return logger


def add_file_handler(file_path: str | Path) -> None:
    """
    Add a file handler to the root logger.

    Parameters
    ----------
    file_path : str or Path
        Path where the logger should write to.

    Raises
    ------
    AttributeError
        When the root logger is not properly initialized (None).
    """

    global _ROOT_LOGGER

    if not isinstance(_ROOT_LOGGER, logging.Logger):
        raise AttributeError(
            "The root logger is not initialized properly. "
            "Did you call `get_root_logger()`? "
            f"Root logger: {_ROOT_LOGGER}"
        )

    # Set file handler
    file_formatter = logging.Formatter(
        "%(asctime)s [%(name)s][%(levelname)s] %(message)s  "
        "<%(filename)s:%(lineno)d> (%(relative)ss)",
        datefmt="%H:%M",
    )
    file_handler = logging.FileHandler(file_path, mode="a")
    file_handler.setLevel(logging.NOTSET)
    file_handler.setFormatter(file_formatter)
    _add_filters(file_handler)
    _ROOT_LOGGER.addHandler(file_handler)


def del_file_handlers() -> None:
    """
    Delete all file handlers in the root logger.

    Raises
    ------
    AttributeError
        When the root logger is not properly initialized (None).
    """
    global _ROOT_LOGGER

    if not isinstance(_ROOT_LOGGER, logging.Logger):
        raise AttributeError(
            "The root logger is not initialized properly. "
            "Did you call `get_root_logger()`? "
            f"Root logger: {_ROOT_LOGGER}"
        )

    for handler in _ROOT_LOGGER.handlers:
        if isinstance(handler, logging.FileHandler):
            _ROOT_LOGGER.removeHandler(handler)


def get_root_logger() -> logging.Logger:
    """
    Get the root logger. If not yet done the logger will be initialized.
    """

    global _ROOT_LOGGER

    # Do not initialize the same logger multiple times
    if isinstance(_ROOT_LOGGER, logging.Logger):
        return _ROOT_LOGGER
    # First time called -> initialize logger
    _ROOT_LOGGER = _create_logger()

    return _ROOT_LOGGER


def del_root_logger() -> None:
    """Reset the root logger and set it to None."""

    global _ROOT_LOGGER
    _ROOT_LOGGER = None
