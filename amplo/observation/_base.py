#  Copyright (c) 2022 by Amplo.

"""
Base class used to build new observers.
"""

import abc
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split

from amplo.utils import check_dtypes

if TYPE_CHECKING:
    from amplo import Pipeline

__all__ = ["BaseObserver", "PipelineObserver", "ProductionWarning", "_report_obs"]


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
    _obs_type : str
        Name of the observation.
    CLASSIFICATION : str
        Name for a classification mode.
    REGRESSION : str
        Name for a regression mode.
    """

    _obs_type = None
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

    def __init__(self, pipeline: "Pipeline"):
        super().__init__()

        if not type(pipeline).__name__ == "Pipeline":
            raise ValueError("Must be an Amplo pipeline.")

        # Set pipeline
        self._pipe = pipeline

        # Split data
        self.xt, self.xv, self.yt, self.yv = train_test_split(
            self.x, self.y, test_size=0.3, random_state=9276306
        )

    @property
    def obs_type(self) -> str:
        """
        Name of the observation type.
        """
        if not self._obs_type or not isinstance(self._obs_type, str):
            raise AttributeError("Class attribute `_obs_type` is not set.")
        return self._obs_type

    @property
    def mode(self):
        return self._pipe.mode

    @property
    def scorer(self):
        return get_scorer(self._pipe.objective)

    @property
    def model(self):
        return deepcopy(self._pipe.best_model)

    @property
    def fitted_model(self):
        fitted_model = self.model
        fitted_model.fit(self.xt, self.yt)
        return fitted_model

    @property
    def x(self):
        # Use best feature set, if available.
        if self._pipe.best_feature_set is not None:
            select_columns = self._pipe.feature_sets[self._pipe.best_feature_set]
            return self._pipe.x[select_columns]
        else:
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

    def report(self: PipelineObserver, *args, **kwargs):
        assert isinstance(self, PipelineObserver)
        status_ok, message = func(self, *args, **kwargs)
        self.report_observation(self.obs_type, func.__name__, status_ok, message)

    return report
