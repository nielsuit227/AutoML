#  Copyright (c) 2022 by Amplo.

from catboost import CatBoostRegressor as _CatBoostRegressor
from sklearn.model_selection import train_test_split

from amplo.regression._base import BaseRegressor
from amplo.utils import check_dtypes


class CatBoostRegressor(BaseRegressor):
    """
    Amplo wrapper for catboost.CatBoostRegressor.

    Parameters
    ----------
    test_size : float, default: 0.1
        Test size for train-test-split in fitting the model.
    random_state : int, default: None
        Random state for train-test-split in fitting the model.
    verbose : {0, 1, 2}, default: 0
        Verbose logging.
    **model_params : Any
        Model parameters for underlying catboost.CatBoostRegressor.
    """

    model: _CatBoostRegressor  # type hint

    def __init__(self, test_size=0.1, random_state=None, verbose=0, **model_params):
        # Verify input dtypes and integrity
        check_dtypes(
            ("test_size", test_size, float),
            ("random_state", random_state, (type(None), int)),
            ("model_params", model_params, dict),
        )
        if not 0 <= test_size < 1:
            raise ValueError(f"Invalid attribute for test_size: {test_size}")

        # Set up model
        default_model_params = {
            "n_estimators": 1000,
            "allow_writing_files": False,
            "early_stopping_rounds": 100,
            "use_best_model": True,
            "verbose": verbose,
        }
        for k, v in default_model_params.items():
            if k not in model_params:
                model_params[k] = v
        model = _CatBoostRegressor(**model_params)

        # Set attributes
        self.test_size = test_size
        self.random_state = random_state

        super().__init__(model=model, verbose=verbose)

    def _fit(self, x, y=None, **fit_params):
        # Split data and fit model
        xt, xv, yt, yv = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(
            xt,
            yt,
            eval_set=[(xv, yv)],
            early_stopping_rounds=self.model.get_params().get("early_stopping_rounds"),
            use_best_model=self.model.get_params().get("use_best_model"),
        )
