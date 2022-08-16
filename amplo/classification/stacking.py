#  Copyright (c) 2022 by Amplo.

import numpy as np
from sklearn.ensemble import StackingClassifier as _StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from amplo.classification._base import BaseClassifier
from amplo.utils import check_dtypes, get_model


def _get_default_estimators(n_samples=None):
    defaults = {
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "GaussianNB": GaussianNB,
        "KNeighborsClassifier": KNeighborsClassifier,
        "LogisticRegression": LogisticRegression,
    }
    if n_samples is not None and n_samples < 5000:
        defaults.update({"SVC": SVC})

    return defaults


def _make_estimator_stack(estimators, add_defaults=True, n_samples=None):
    """
    Make a stack of estimators for the stacking model.

    Parameters
    ----------
    estimators : list of str
        List of estimators for the stack.
    add_defaults : bool, default: True
        Whether to add default estimators to the stack.
    n_samples : int, optional
        (Expected) number of samples to determine the default estimators.

    Returns
    -------
    list of (str, estimator)
        Stack of estimators.
    """
    check_dtypes(
        ("estimators", estimators, list),
        *[(f"estimators_item: `{est}`", est, str) for est in estimators],
    )

    # Initialize
    stack = {}

    # Add default models
    if add_defaults:
        for model_name, model in _get_default_estimators(n_samples).items():
            stack[model_name] = model()  # initialize model

    # Add Amplo models
    for model_name in estimators:
        if model_name in stack:
            # Skip default models
            continue
        stack[model_name] = get_model(
            model_name, mode="classification", samples=n_samples
        )

    return [(key, value) for key, value in stack.items()]


def _get_final_estimator(n_samples=None, n_features=None):
    check_dtypes(
        ("n_samples", n_samples, (type(None), int)),
        ("n_features", n_features, (type(None), int)),
    )

    many_samples = not n_samples or n_samples > 10_000
    many_features = not n_features or n_features > 100

    solver = "lbfgs" if many_samples or many_features else "sag"
    return LogisticRegression(max_iter=2000, solver=solver)


class StackingClassifier(BaseClassifier):
    """
    Stacking classifier.

    Parameters
    ----------
    add_to_stack : list of str, optional
        List of estimators for the stack of estimators.
    add_defaults_to_stack : bool, default: True
        Whether to add default estimators to the stack. This option will be set to True
        when the `add_to_stack` parameter is None.
    n_samples : int, optional
        (Expected) number of samples.
    n_features : int, optional
        (Expected) number of features.
    verbose : {0, 1, 2}, default: 0
        Verbose logging.
    **model_params : Any
        Model parameters for underlying models.
    """

    _add_to_settings = ["_mean", "_std", *BaseClassifier._add_to_settings]
    model: _StackingClassifier  # type hint

    def __init__(
        self,
        add_to_stack=None,
        add_defaults_to_stack=True,
        n_samples=None,
        n_features=None,
        verbose=0,
        **model_params,
    ):
        check_dtypes(("add_defaults_to_stack", add_defaults_to_stack, bool))

        # Set attributes
        if add_to_stack is None:
            add_to_stack = []
            add_defaults_to_stack = True
        model = _StackingClassifier(
            _make_estimator_stack(add_to_stack, add_defaults_to_stack, n_samples),
            _get_final_estimator(n_samples, n_features),
        )
        model.set_params(**model_params)

        # Set attributes
        self.add_to_stack = add_to_stack
        self.add_defaults_to_stack = add_defaults_to_stack
        self.n_samples = n_samples
        self.n_features = n_features

        super().__init__(model=model, verbose=verbose)

    def _fit(self, x, y=None, **fit_params):
        # When `self.n_samples` or `self.n_features` is None or badly initialized, we
        # reset the stacking estimator as its stack and final estimator depend on that.
        if self.n_samples != x.shape[0] or self.n_features != x.shape[1]:
            self.n_samples, self.n_features = x.shape

            # Get previous model parameters
            prev_model_params = self._get_model_params()

            # Init new stacking classifier
            self.model = _StackingClassifier(
                _make_estimator_stack(
                    self.add_to_stack or [], self.add_defaults_to_stack, self.n_samples
                ),
                _get_final_estimator(self.n_samples, self.n_features),
            )

            # Update model parameters from previous model
            model_params = self._get_model_params()
            for key in set(model_params).intersection(prev_model_params):
                model_params[key] = prev_model_params[key]
            self.model.set_params(**model_params)

        # Normalize
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std[std == 0] = 1
        self._mean = np.asarray(mean).reshape(-1).tolist()
        self._std = np.asarray(std).reshape(-1).tolist()
        x -= mean
        x /= std

        # Fit model
        self.model.fit(x, y)

    def _predict(self, x, y=None, **kwargs):
        mean = np.array(self._mean)
        std = np.array(self._std)
        return self.model.predict((x - mean) / std, **kwargs).reshape(-1)

    def predict_proba(self, x, **kwargs):
        self.check_is_fitted()
        mean = np.array(self._mean)
        std = np.array(self._std)
        return self.model.predict_proba((x - mean) / std, **kwargs)

    def _get_model_params(self, deep=True):
        model_params = self.model.get_params(deep)

        non_serializable = ["estimators", "final_estimator"]
        if deep:
            non_serializable.extend(name for name, _ in model_params["estimators"])
        for key in non_serializable:
            model_params.pop(key)

        return model_params
