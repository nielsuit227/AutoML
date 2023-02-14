#  Copyright (c) 2022 by Amplo.

"""
Implements base classes.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from warnings import warn

import numpy.typing as npt
import pandas as pd
from sklearn.base import clone
from typing_extensions import Self

from amplo.base.exceptions import NotFittedError
from amplo.utils.logging import get_root_logger
from amplo.utils.util import check_dtype, check_dtypes

__all__ = [
    "AmploObject",
    "BaseEstimator",
    "BaseTransformer",
    "LoggingMixin",
    "Result",
]


class AmploObject:
    """
    Base class for Amplo objects.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @classmethod
    def _get_param_names(cls) -> list[str]:
        # copied from sklearn.base.BaseEstimator
        """
        Get parameter names for the object.
        """
        # Introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(cls.__init__)
        # Consider the positional constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    f"Amplo objects should always specify their parameters in the "
                    f"signature of their __init__ (no varargs). {cls} with constructor "
                    f"{init_signature} doesn't follow this convention."
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        # copied from sklearn.base.BaseEstimator
        """
        Get parameters for this object.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this and contained subobjects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out: dict[str, Any] = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params) -> Self:
        # copied from sklearn.base.BaseEstimator
        """
        Set the parameters of this object.

        The method works on simple objects as well as on nested objects.
        The latter have parameters of the form '<component>__<parameter>' so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self
            Amplo object instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params: defaultdict[str, dict[str, str]]
        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def reset(self) -> Self:
        # copied from sklearn.base.BaseEstimator
        """
        Reset the object to a clean post-init state.

        Equivalent to sklearn.clone but overwrites self.
        After self.reset() call, self is equal in value to
        `type(self)(**self.get_params(deep=False))`

        Detail behaviour:
            1. removes any object attributes, except:
                - hyperparameters = arguments of __init__
                - object attributes containing double-underscores, i.e. "__"
            2. runs __init__ with current values of hyperparameters (result of
            get_params)

        Not affected by the reset are:
        - object attributes containing double-underscores
        - class and object methods, class attributes
        """
        # retrieve parameters to copy them later
        params = self.get_params(deep=False)

        # delete all object attributes in self
        attrs = [attr for attr in dir(self) if "__" not in attr]
        cls_attrs = [attr for attr in dir(type(self))]
        self_attrs = set(attrs).difference(cls_attrs)
        for attr in self_attrs:
            delattr(self, attr)

        # run init with a copy of parameters self had at the start
        self.__init__(**params)  # type: ignore[misc]

        return self

    def clone(self, *, safe: bool = True) -> Self:
        """
        Constructs a new unfitted object with the same parameters.

        Clone does a deep copy of the model in an object without actually copying
        attached data. It yields a new object with the same parameters that has not been
        fitted on any data.

        Parameters
        ----------
        safe : bool, default=True
            If safe is False, clone will fall back to a deep copy on objects
            that are not estimators.
        """
        return clone(self, safe=safe)


class BaseEstimator(AmploObject, metaclass=ABCMeta):
    """
    Estimator base class.

    Extends the AmploObject class with an is_fitted attribute.

    Attributes
    ----------
    is_fitted_ : bool
        Indicates whether the estimator is fitted.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model: Any
        self.is_fitted_ = False
        self.classes_: npt.NDArray[Any] | None = None

    @abstractmethod
    def fit(self, *args, **kwargs) -> Any:
        pass

    def score(self, x: pd.DataFrame, y: pd.Series | None, *args, **kwargs) -> Any:
        return self.model.score(x, y, *args, **kwargs)

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        pass

    def predict_proba(self, x: pd.DataFrame, *args, **kwargs) -> Any:
        if not self.is_fitted_:
            raise NotFittedError
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Model has no attribute `predict_proba`.")

        return self.model.predict_proba(x, *args, **kwargs)


class BaseTransformer(AmploObject, metaclass=ABCMeta):
    """
    Transformer base class.
    """

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> Any:
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class LoggingMixin:
    """
    Mixin class for adding logging capability to an object.

    Parameters
    ----------
    verbose : int
        Verbosity for logger.

    Notes
    -----
    The logging level depends on the parameter verbose as follows:
        - verbose=0: warnings or higher priority
        - verbose=1: info or higher priority
        - verbose=2: debugging info or higher priority
    """

    def __init__(self, verbose=0):
        check_dtype("verbose", verbose, (float, int))

        # Set logging level based on verbose
        if verbose < 0:
            warn("Parameter 'verbose' cannot be smaller than zero.")
            verbose = 0
            logging_level = logging.WARNING
        elif verbose == 0:
            logging_level = logging.WARNING
        elif verbose == 1:
            logging_level = logging.INFO
        else:  # verbose >= 2
            logging_level = logging.DEBUG

        self.verbose = verbose
        # NOTE: Without creating a new logger (through `getChild`), setting the
        #  logging level will influence all logging. Setting logging levels individually
        #  per class is therefore not possible.
        self.logger = get_root_logger().getChild(self.__class__.__name__)
        self.logger.setLevel(logging_level)


@dataclass
class Result:
    date: str
    model: str
    params: dict[str, Any]
    feature_set: str
    score: float
    worst_case: float
    time: float

    def __post_init__(self):
        check_dtypes(
            ("date", self.date, str),
            ("model", self.model, str),
            ("params", self.params, dict),
            ("feature_set", self.feature_set, str),
            ("score", self.score, float),
            ("worst_case", self.worst_case, float),
            ("time", self.time, float),
        )

    def __lt__(self, other):
        return self.worst_case < other.worst_case
