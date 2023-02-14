#  Copyright (c) 2022 by Amplo.

"""
Base class used to build new observers.
"""

from __future__ import annotations

import abc
import warnings
from typing import Callable

import numpy as np

from amplo.utils import check_dtypes

__all__ = ["BaseObserver", "ProductionWarning", "_report_obs"]


class ProductionWarning(RuntimeWarning):
    """
    Warning for suspicions before moving to production.
    """


class BaseObserver(abc.ABC):
    """
    Abstract base class to build new observers.

    Subclass this class.

    Attributes
    ----------
    observations : list of dict
        A list of observations.  Each observation is a dictionary containing the
        keys `type` (str), `name` (str), `status_ok` (bool) and `description`
        (str) - with corresponding dtypes.
    """

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    _obs_type: str | None = None

    def __init__(self):
        self.observations: list[dict[str, str | bool]] = []

    def report_observation(self, typ, name, status_ok, message):
        """
        Report an observation to the observer.

        An observation will trigger a warning when `status_ok` is false.

        Parameters
        ----------
        typ : str
            Observation type.
        name : str
            Observation name.
        status_ok : bool
            Observation status. If false, a warning will be triggered.
        message : str
            A brief description of the observation and its results.
        """
        # Check input
        check_dtypes(
            ("typ", typ, str),
            ("name", name, str),
            ("status_ok", status_ok, (bool, np.bool_)),
            ("message", message, str),
        )
        if not isinstance(status_ok, bool):
            status_ok = bool(status_ok)

        # Trigger warning when status is not okay
        if not status_ok:
            msg = (
                "A production observation needs inspection. Please evaluate "
                f"why a warning was triggered from `{typ}/{name}`. "
                f"Warning message: {message}"
            )
            warnings.warn(ProductionWarning(msg))

        # Add observation to list
        obs = {"typ": typ, "name": name, "status_ok": status_ok, "message": message}
        self.observations.append(obs)

    @property
    def obs_type(self) -> str:
        """
        Name of the observation type.
        """
        if not self._obs_type or not isinstance(self._obs_type, str):
            raise AttributeError("Class attribute `_obs_type` is not set.")
        return self._obs_type

    @abc.abstractmethod
    def observe(self, *args, **kwargs):
        """
        Observe the data, model, ...

        Observations should be reported via `self.report_observation()`.
        """


def _report_obs(func: Callable[..., tuple[bool, str]]) -> Callable[..., None]:
    """
    Decorator for checker function in observer class.

    Parameters
    ----------
    func : function
        The class method that shall report an observation. It must return the
        observation status (bool) and its message (str).

    Returns
    -------
    decorator
    """

    def report(self: BaseObserver, *args, **kwargs):
        status_ok, message = func(self, *args, **kwargs)
        self.report_observation(self.obs_type, func.__name__, status_ok, message)

    return report
