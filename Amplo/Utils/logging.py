from pathlib import Path
import logging
import colorlog


__all__ = ['get_logger']


_nameToLevel = {
    'CRITICAL': logging.CRITICAL,
    'FATAL': logging.FATAL,
    'ERROR': logging.ERROR,
    'WARN': logging.WARNING,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET,
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


def get_logger(name='AutoML', log_path=None, level=logging.NOTSET, capture_warnings=True) -> logging.Logger:
    """
    Create a logging Logger

    Parameters
    ----------
    name : str
        Name of the logger
    log_path : str or Path, optional
        Specifies the logging path, in case you want to store the logs in a file.
    level : int or str, optional
        Logging level
    capture_warnings : bool
        Whether to capture `warnings.warn(...)` with the logger.
        Note that this option is set globally!

    Returns
    -------
    logging.Logger
    """
    # TODO: Add option `parent` to use `parent.getChild()`
    #  -> This is part of https://amplo.atlassian.net/browse/AML-71

    # Safety check for logging level
    level = _check_logging_level(level)

    # Get custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Set console handler
    console_formatter = colorlog.ColoredFormatter(
        '%(white)s[%(name)s] %(log_color)s%(levelname)s: %(message)s %(white)s<%(filename)s:%(lineno)d>')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Set file handler
    if log_path is not None:
        if Path(log_path).suffix != '.log':
            logger.warning('It is recommended naming a log file as `*.log`')
        file_formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Capture warnings
    if capture_warnings:
        # Capture warnings from `warnings.warn(...)`
        logging.captureWarnings(True)
        # Get py-warnings logger and add handler
        py_warnings_logger = logging.getLogger('py.warnings')
        py_warnings_logger.addHandler(console_handler)

    return logger
