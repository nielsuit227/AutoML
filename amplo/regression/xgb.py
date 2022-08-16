#  Copyright (c) 2022 by Amplo.

import xgboost.callback
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor as _XGBRegressor

from amplo.regression._base import BaseRegressor
from amplo.utils import check_dtypes


def _validate_xgboost_callbacks(callbacks):
    if not callbacks:
        return []

    for cb in callbacks:
        raise NotImplementedError

    return callbacks


class XGBRegressor(BaseRegressor):
    """
    Amplo wrapper for xgboost.XGBRegressor.

    Parameters
    ----------
    callbacks : list of str, optional
        ...
    test_size : float, default: 0.1
        Test size for train-test-split in fitting the model.
    early_stopping_rounds : int, default: 100
        Number of early stopping rounds.
    random_state : int, default: None
        Random state for train-test-split in fitting the model.
    verbose : {0, 1, 2}, default: 0
        Verbose logging.
    **model_params : Any
        Model parameters for underlying xgboost.XGBRegressor.
    """

    model: _XGBRegressor  # type hint

    def __init__(
        self,
        callbacks=None,
        test_size=0.1,
        early_stopping_rounds=None,
        random_state=None,
        verbose=0,
        **model_params,
    ):
        # Verify input dtypes and integrity
        check_dtypes(
            ("callbacks", callbacks, (type(None), list)),
            ("test_size", test_size, float),
            ("random_state", random_state, (type(None), int)),
            ("early_stopping_rounds", early_stopping_rounds, (type(None), int)),
            ("model_params", model_params, dict),
        )
        if not 0 <= test_size < 1:
            raise ValueError(f"Invalid attribute for test_size: {test_size}")

        # Set attributes
        self.callbacks = callbacks
        self.test_size = test_size
        self.early_stopping_rounds = early_stopping_rounds or 100
        self.random_state = random_state

        # Set up model
        default_model_params = {
            "n_estimators": 100,  # number of boosting rounds
            "random_state": random_state,
            "use_label_encoder": False,  # is deprecated
            "verbosity": verbose,
        }
        for k, v in default_model_params.items():
            if k not in model_params:
                model_params[k] = v
        model = _XGBRegressor(**model_params)

        super().__init__(model=model, verbose=verbose)

    def _fit(self, x, y=None, **fit_params):
        # Set up fitting callbacks
        callbacks = _validate_xgboost_callbacks(self.callbacks)
        callbacks.append(xgboost.callback.EarlyStopping(self.early_stopping_rounds))

        # Split data and fit model
        xt, xv, yt, yv = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(
            xt, yt, eval_set=[(xv, yv)], callbacks=callbacks, verbose=bool(self.verbose)
        )
