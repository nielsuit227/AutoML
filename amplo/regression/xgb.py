#  Copyright (c) 2022 by Amplo.

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor as _XGBRegressor

from amplo.classification.xgb import _validate_xgboost_callbacks
from amplo.regression._base import BaseRegressor
from amplo.utils import check_dtypes


class XGBRegressor(BaseRegressor):
    """
    Amplo wrapper for xgboost.XGBRegressor.

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
        Model parameters for underlying xgboost.XGBRegressor.
    """

    model: _XGBRegressor  # type hint

    def __init__(
        self,
        callbacks=None,
        test_size=0.1,
        random_state=None,
        verbose=0,
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
        model = _XGBRegressor(
            **model_params, callbacks=_validate_xgboost_callbacks(callbacks)
        )

        # Set attributes
        self.callbacks = callbacks
        self.test_size = test_size
        self.random_state = random_state

        super().__init__(model=model, verbose=verbose)

    def fit(self, x, y=None, **fit_params):
        # Split data and fit model
        xt, xv, yt, yv = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )
        # Note that we have to set `verbose` in `fit`.
        # Otherwise, it will still verbose print the evaluation.
        self.model.fit(xt, yt, eval_set=[(xv, yv)], verbose=bool(self.verbose))

        self.is_fitted_ = True
        return self
