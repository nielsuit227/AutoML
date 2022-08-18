#  Copyright (c) 2022 by Amplo.

"""
Implements base classes.
"""

import importlib
import inspect
import logging
from collections import defaultdict
from warnings import warn

from amplo.base.exceptions import NotFittedError
from amplo.utils.logging import logger

__all__ = [
    "BaseObject",
    "BaseEstimator",
    "BaseTransformer",
    "BasePredictor",
    "LoggingMixin",
]


class BaseObject:
    # This class is strongly inspired by sklearn's BaseEstimator.
    """
    Object base class.

    Class attributes
    ----------------
    _add_to_settings : list of str
        Attribute names to be included in settings.
    """

    _add_to_settings = []

    def get_settings(self, deep=True):
        """
        Get setting parameters for this object.

        Parameters
        ----------
        deep : bool
            If True, will return the parameters for this object and contained
            sub-objects.

        Returns
        -------
        settings : dict
            Settings for this object.
        """
        out = {"<init_params>": self.get_params(deep=False)}
        for key in self._add_to_settings:
            value = getattr(self, key, None)
            if deep and isinstance(value, BaseObject):
                deep_items = value.get_settings().items()
                out.update((f"{key}__{k}", val) for k, val in deep_items)
                out[key] = str(type(value))
            else:
                out[key] = value
        return out

    def load_settings(self, settings):
        """
        Load settings for this object.

        Parameters
        ----------
        settings : dict

        Returns
        -------
        self :
            An object with injected settings.
        """
        if not settings:
            return self
        settings = settings.copy()
        valid_settings = self.get_settings(deep=True)

        if "<init_params>" in settings:
            self.__init__(**settings.pop("<init_params>"))  # noqa

        nested_settings = defaultdict(dict)  # grouped by prefix
        for key, value in settings.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_settings:
                local_valid_params = self._add_to_settings
                raise ValueError(
                    f"Invalid parameter {key!r} for object {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_settings[key][sub_key] = value
            else:
                if isinstance(value, str) and "<class " in value:
                    _, cls, _ = value.split("'")
                    module_name, class_name = cls.rsplit(".", 1)
                    module = importlib.import_module(module_name)
                    init_params = settings.get(f"{key}__<init_params>", {})
                    value = getattr(module, class_name)(**init_params)
                setattr(self, key, value)
                valid_settings[key] = value

        for key, sub_params in nested_settings.items():
            valid_settings[key].load_settings(sub_params)

        return self

    @classmethod
    def _get_param_names(cls):
        """
        Get parameter names for the estimator.
        """
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

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

    def reset(self):
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
        self.__init__(**params)  # noqa

        return self


class BaseEstimator(BaseObject):
    """
    Estimator base class.

    Extends the BaseObject class with an is_fitted attribute.

    Attributes
    ----------
    _is_fitted : bool
        Indicates whether the estimator is fitted.
    """

    _add_to_settings = ["_is_fitted", *BaseObject._add_to_settings]

    def __init__(self):
        self._is_fitted = False
        super().__init__()

    @property
    def is_fitted(self):
        """Whether `fit` has been called."""
        # Note: for backwards compatibility
        if hasattr(self, "_is_fitted"):
            return self._is_fitted
        else:
            return True

    def check_is_fitted(self):
        """
        Asserts that estimator is fitted.

        Raises
        ------
        NotFittedError
            When estimator is not fitted.
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not been "
                f"fitted yet; please call `fit` first."
            )

    def fit(self, x, y=None, **fit_params):
        """
        Fit the estimator to x and optionally to y.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets is_fitted flag to True.
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame
            Feature data to fit.
        y : numpy.ndarray or pandas.Series
            Target data to fit.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        self : estimator
            A fitted instance of the estimator.
        """
        # If fit is called, estimator is reset, including fitted state
        self.reset()

        # Pass to inner fit
        self._fit(x, y, **fit_params)

        # This should happen last: fitted state is set to True
        self._is_fitted = True

        return self

    def _fit(self, x, y=None, **fit_params):
        """
        Fit the estimator to x and optionally to y.

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame
            Checked feature data to fit.
        y : numpy.ndarray or pandas.Series
            Checked target data to fit.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        self : estimator
            A fitted instance of the transformer.
        """
        return self


class BaseTransformer(BaseEstimator):
    """
    Transformer base class.
    """

    def transform(self, x, y=None):
        """
        Transform data and return it.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self._is_fitted

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame
            Feature data to transform.
        y : numpy.ndarray or pandas.Series
            Additional target data to transform.

        Returns
        -------
        pandas.DataFrame
            Transformed version of x.
        """
        # Check whether is fitted
        self.check_is_fitted()

        # Transform data
        xt = self._transform(x, y)

        return xt

    def fit_transform(self, x, y=None, **fit_params):
        """
        Fit and transform data.

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame
            Feature data to transform.
        y : numpy.ndarray or pandas.Series
            Additional target data to transform.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        pandas.DataFrame
            Transformed version of x.
        """
        # Default, non-optimized version for `fit_transform`. Overwrite, when
        # can be optimized.
        return self.fit(x, y, **fit_params).transform(x, y)

    def _transform(self, x, y=None):
        """
        Transform data.

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame
            Checked feature data to fit.
        y : numpy.ndarray or pandas.Series
            Checked target data to fit.

        Returns
        -------
        pandas.DataFrame
            Transformed version of x.
        """
        raise NotImplementedError("Abstract method.")


class BasePredictor(BaseEstimator):
    """
    Predictor base class.
    """

    def predict(self, x, y=None, **predict_params):
        """
        Transform data and return it.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self._is_fitted

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame
            Feature data to transform.
        y : numpy.ndarray or pandas.Series
            Additional target data to transform.
        **predict_params
            Additional predict parameters.

        Returns
        -------
        pandas.DataFrame
            Prediction for x.
        """
        # Check whether is fitted
        self.check_is_fitted()

        # Transform data
        xt = self._predict(x, y, **predict_params)

        return xt

    def fit_predict(self, x, y=None, **fit_predict_params):
        """
        Fit and predict data.

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame
            Feature data to transform.
        y : numpy.ndarray or pandas.Series
            Additional target data to transform.
        **fit_predict_params : dict
            Additional fit and predict parameters.

        Returns
        -------
        pandas.DataFrame
            Prediction for x.
        """
        # Default, non-optimized version for `fit_predict`. Overwrite, when
        # can be optimized.
        return self.fit(x, y, **fit_predict_params).predict(x, **fit_predict_params)

    def _predict(self, x, y=None, **predict_params):
        """
        Predict on data.

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame
            Checked feature data to fit.
        y : numpy.ndarray or pandas.Series
            Checked target data to fit.
        **predict_params : dict
            Additional predict parameters.

        Returns
        -------
        pandas.DataFrame
            Prediction for x.
        """
        raise NotImplementedError("Abstract method.")


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
        if not isinstance(verbose, (float, int)):
            raise ValueError(f"Invalid dtype for `verbose`: {type(verbose)}.")

        # Set logging level based on verbose
        if verbose < 0:
            warn("`verbose` cannot be smaller than zero.", UserWarning)
            verbose = 0
            logging_level = logging.WARNING
        elif verbose == 0:
            logging_level = logging.WARNING
        elif verbose == 1:
            logging_level = logging.INFO
        else:  # verbose >= 2
            logging_level = logging.DEBUG

        self.verbose = verbose
        # Notice: Without creating a new logger (through `getChild`), setting the
        # logging level will influence all logging. Setting logging levels individually
        # per class is therefore not possible.
        self.logger = logger.getChild(type(self).__name__)
        self.logger.setLevel(logging_level)
