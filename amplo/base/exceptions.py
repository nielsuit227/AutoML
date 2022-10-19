#  Copyright (c) 2022 by Amplo.

"""
Base exceptions.
"""

__all__ = [
    "ExperimentalWarning",
    "NotFittedError",
]


# --------------------------------------------------------------------------
# Warnings


class ExperimentalWarning(UserWarning):
    """
    Warning for experimental features.
    """


# --------------------------------------------------------------------------
# Errors


class EmptyFileError(Exception):
    """
    File contains no data.
    """


class NotFittedError(Exception):
    """
    Object is not fitted.
    """
