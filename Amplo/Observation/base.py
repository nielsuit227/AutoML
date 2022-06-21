# Copyright by Amplo
"""
Base class used to build new observers.
"""

import abc
from copy import deepcopy
from typing import Dict, List, Union, TYPE_CHECKING
import warnings

import numpy as np
from sklearn.metrics import get_scorer

if TYPE_CHECKING:
    from Amplo import Pipeline

__all__ = ["BaseObserver", "PipelineObserver", "ProductionWarning",
           "_report_obs"]


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

    def __init__(self):
        self.observations: List[Dict[str, Union[str, bool]]] = []

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
        if not isinstance(typ, str):
            raise ValueError("Invalid dtype for observation type.")
        if not isinstance(name, str):
            raise ValueError("Invalid dtype for observation name.")
        if not isinstance(status_ok, (bool, np.bool_)):
            raise ValueError("Invalid dtype for observation status.")
        if not isinstance(message, str):
            raise ValueError("Invalid dtype for observation message.")

        # Trigger warning when status is not okay
        if not status_ok:
            msg = ("A production observation needs inspection. Please evaluate "
                   f"why a warning was triggered from `{typ}/{name}`. "
                   f"Warning message: {message}")
            warnings.warn(ProductionWarning(msg))

        # Add observation to list
        obs = {"typ": typ, "name": name, "status_ok": status_ok,
               "message": message}
        self.observations.append(obs)

    @abc.abstractmethod
    def observe(self):
        """
        Observe the data, model, ...

        Observations should be reported via `self.report_observation()`.
        """


class PipelineObserver(BaseObserver, metaclass=abc.ABCMeta):
    """
    Extension of ``BaseObserver``.

    Unifies behavior of class initialization.

    Parameters
    ----------
    pipeline : Pipeline
        The amplo pipeline object that will be observed.

    Class Attributes
    ----------------
    TYPE : str
        Name of the observation.
    CLASSIFICATION : str
        Name for a classification mode.
    REGRESSION : str
        Name for a regression mode.
    """

    TYPE = None
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

    def __init__(self, pipeline: 'Pipeline'):
        super().__init__()

        if not type(pipeline).__name__ == "Pipeline":
            raise ValueError("Must be an Amplo pipeline.")

        self._pipe = pipeline

    @property
    def obs_type(self) -> str:
        """
        Name of the observation type.
        """
        if not self.TYPE or not isinstance(self.TYPE, str):
            raise AttributeError("Class attribute `TYPE` is not set.")
        return self.TYPE

    @property
    def mode(self):
        return self._pipe.mode

    @property
    def model(self):
        return deepcopy(self._pipe.best_model)

    @property
    def scorer(self):
        return get_scorer(self._pipe.objective)

    @property
    def x(self):
        return self._pipe.x

    @property
    def y(self):
        return self._pipe.y


def _report_obs(func):
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

    def report(self: PipelineObserver):
        assert isinstance(self, PipelineObserver)
        status_ok, message = func(self)
        self.report_observation(self.obs_type, func.__name__, status_ok, message)

    return report
