#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost.callback
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier as _XGBClassifier

from amplo.classification._base import BaseClassifier
from amplo.utils import check_dtypes


def _validate_xgboost_callbacks(callbacks):
    if not callbacks:
        return []

    valild_callbacks = []
    for cb in callbacks:
        if not isinstance(cb, str):
            raise ValueError(f"Expected a string but got '{cb}' of type '{type(cb)}'.")

        if cb.startswith("early_stopping_rounds="):
            n_rounds = int(cb.removeprefix("early_stopping_rounds="))
            valild_callbacks.append(xgboost.callback.EarlyStopping(n_rounds))

        else:
            raise NotImplementedError(f"Unknown callback '{cb}'.")

    return valild_callbacks


class XGBClassifier(BaseClassifier):
    """
    Amplo wrapper for xgboost.XGBClassifier.

    Parameters
    ----------
    callbacks : list of str, optional
        The following callbacks are currently supported:
            - early stopping, "early_stopping_rounds=100"
    test_size : float, default: 0.1
        Test size for train-test-split in fitting the model.
    random_state : int, default: None
        Random state for train-test-split in fitting the model.
    verbose : {0, 1, 2}, default: 0
        Verbose logging.
    **model_params : Any
        Model parameters for underlying xgboost.XGBClassifier.
    """

    model: _XGBClassifier  # type hint

    def __init__(
        self,
        callbacks: list[str] | None = None,
        test_size: float = 0.1,
        random_state: int | None = None,
        verbose: int = 0,
        **model_params,
    ):
        # Verify input dtypes and integrity
        check_dtypes(
            ("callbacks", callbacks, (type(None), list)),
            ("test_size", test_size, float),
            ("random_state", random_state, (type(None), int)),
            ("model_params", model_params, dict),
        )
        if not 0 <= test_size < 1:
            raise ValueError(f"Invalid attribute for test_size: {test_size}")

        # Set up callbacks
        callbacks = callbacks or []
        for cb_name, cb_default_value in [("early_stopping_rounds", 100)]:
            # Skip if already present in callbacks
            if any(callback.startswith(cb_name) for callback in callbacks):
                continue
            # Pop model parameters into callbacks
            callbacks.append(f"{cb_name}={model_params.pop(cb_name, cb_default_value)}")

        # Set up model
        default_model_params = {
            "n_estimators": 100,  # number of boosting rounds
            "random_state": random_state,
            "verbosity": verbose,
        }
        for k, v in default_model_params.items():
            if k not in model_params:
                model_params[k] = v
        model = _XGBClassifier(
            **model_params, callbacks=_validate_xgboost_callbacks(callbacks)
        )

        # Set attributes
        self.callbacks = callbacks
        self.test_size = test_size
        self.random_state = random_state
        super().__init__(model=model, verbose=verbose)

    def fit(self, x: pd.DataFrame, y: pd.Series, **fit_params):
        self.reset()
        self.classes_ = np.unique(y)
        # Split data and fit model
        xt, xv, yt, yv = train_test_split(
            x, y, stratify=y, test_size=self.test_size, random_state=self.random_state
        )
        # Note that we have to set `verbose` in `fit`.
        # Otherwise, it will still verbose print the evaluation.
        self.model.fit(xt, yt, eval_set=[(xv, yv)], verbose=bool(self.verbose))
        self.is_fitted_ = True
        return self
