#  Copyright (c) 2022 by Amplo.

import lightgbm
from lightgbm import LGBMClassifier as _LGBMClassifier
from sklearn.model_selection import train_test_split

from amplo.classification._base import BaseClassifier
from amplo.utils import check_dtypes


def _validate_lightgbm_callbacks(callbacks):
    if not callbacks:
        return []

    for cb in callbacks:
        raise NotImplementedError

    return callbacks


class LGBMClassifier(BaseClassifier):
    """
    Amplo wrapper for lightgbm.LGBMClassifier.

    Parameters
    ----------
    callbacks : list of str, optional
        ...
    test_size : float, default: 0.1
        Test size for train-test-split in fitting the model.
    random_state : int, default: None
        Random state for train-test-split in fitting the model.
    verbose : {0, 1, 2}, default: 0
        Verbose logging.
    **model_params : Any
        Model parameters for underlying lightgbm.LGBMClassifier.
    """

    model: _LGBMClassifier  # type hint

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

        # Set up model parameters
        default_model_params = {
            "num_iterations": 1000,  # number of boosting rounds
            "force_col_wise": True,  # reduce memory cost
            "early_stopping_rounds": 100,
            "verbose": verbose,
        }
        for k, v in default_model_params.items():
            if k not in model_params:
                model_params[k] = v
        model = _LGBMClassifier(**model_params)

        # Set attributes
        self.callbacks = callbacks
        self.test_size = test_size
        self.random_state = random_state

        super().__init__(model=model, verbose=verbose)

    def _fit(self, x, y=None, **fit_params):
        # Set up fitting callbacks
        callbacks = _validate_lightgbm_callbacks(self.callbacks)
        callbacks.append(
            lightgbm.early_stopping(
                self.model.get_params().get("early_stopping_rounds", 100),
                verbose=False,
            )
        )

        # Split data and fit model
        xt, xv, yt, yv = train_test_split(
            x, y, stratify=y, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(
            xt, yt, eval_set=[(xv, yv)], callbacks=callbacks, eval_metric="logloss"
        )
