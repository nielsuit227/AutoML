#  Copyright (c) 2022 by Amplo.
import logging
from datetime import datetime

import colorlog

__all__ = ["logger"]


_nameToLevel = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.FATAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def _check_logging_level(level):
    """
    Check input whether it is a valid logging level

    Parameters
    ----------
    level : int or str
        Logging level

    Returns
    -------
    int
        (valid) logging level

    Raises
    ------
    ValueError
        If the given logging level is not valid
    """
    # Inspired by logging/__init__.py
    if isinstance(level, int):
        if level not in set(_nameToLevel.values()):
            raise ValueError("Unknown level: %r" % level)
        rv = level
    elif str(level) == level:
        if level not in _nameToLevel:
            raise ValueError("Unknown level: %r" % level)
        rv = _nameToLevel[level]
    else:
        raise TypeError("Level not an integer or a valid string: %r" % level)
    return rv


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


# Get custom logger
logger = logging.getLogger("AmploML")
logger.setLevel("INFO")

# Set console handler
console_formatter = colorlog.ColoredFormatter(
    "%(blue)s[%(name)s]%(log_color)s[%(levelname)s] %(white)s%(message)s "
    "%(light_black)s<%(filename)s:%(lineno)d> (%(relative)ss)"
)
console_handler = logging.StreamHandler()
console_handler.setLevel("INFO")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Set file handler
file_formatter = logging.Formatter(
    "[%(name)s][%(levelname)s] %(message)s <%(filename)s:%(lineno)d> (%(relative)ss)"
)
file_handler = logging.FileHandler("AutoML.log", mode="a")
file_handler.setLevel("INFO")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Add TimeFilter
[handler.addFilter(TimeFilter()) for handler in logger.handlers]

# Capture warnings from `warnings.warn(...)`
logging.captureWarnings(True)
py_warnings_logger = logging.getLogger("py.warnings")
warnings_formatter = colorlog.ColoredFormatter(
    "%(white)s[%(name)s] %(log_color)s%(levelname)s: %(message)s"
)
warnings_handler = logging.StreamHandler()
warnings_handler.setLevel("WARNING")
warnings_handler.setFormatter(warnings_formatter)
py_warnings_logger.addHandler(warnings_handler)


def remove_file_handler():
    logger.removeHandler(file_handler)
