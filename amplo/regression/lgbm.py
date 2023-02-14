#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

from lightgbm import LGBMRegressor as _LGBMRegressor
from sklearn.model_selection import train_test_split

from amplo.classification.lgbm import _validate_lightgbm_callbacks
from amplo.regression._base import BaseRegressor
from amplo.utils import check_dtypes


class LGBMRegressor(BaseRegressor):
    """
    Amplo wrapper for lightgbm.LGBMRegressor.

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
        Model parameters for underlying lightgbm.LGBMRegressor.
    """

    model: _LGBMRegressor  # type hint

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

        # Set up model parameters
        default_model_params = {
            "n_estimators": 1000,  # number of boosting rounds
            "force_col_wise": True,  # reduce memory cost
            "verbosity": verbose - 1,  # don't use "verbose" due to self.reset()
        }
        for k, v in default_model_params.items():
            if k not in model_params:
                model_params[k] = v
        model = _LGBMRegressor(**model_params)

        # Set attributes
        self.callbacks = callbacks
        self.test_size = test_size
        self.random_state = random_state

        super().__init__(model=model, verbose=verbose)

    def fit(self, x, y=None, **fit_params):
        # Set up fitting callbacks
        callbacks = _validate_lightgbm_callbacks(self.callbacks)

        # Split data and fit model
        xt, xv, yt, yv = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(
            xt, yt, eval_set=[(xv, yv)], callbacks=callbacks, eval_metric="rmse"
        )
        self.is_fitted_ = True
        return self
