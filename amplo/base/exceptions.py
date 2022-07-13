#  Copyright (c) 2022 by Amplo.

"""
Base exceptions.
"""

__all__ = [
    "ExperimentalWarning",
    "NotFittedError",
]


class ExperimentalWarning(UserWarning):
    """
    Warning for experimental features.
    """


class NotFittedError(ValueError, AttributeError):
    """
    Exception class to raise if estimator is used before fitting.
    """
