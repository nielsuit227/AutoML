from copy import deepcopy

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


class LGBMRegressor:
    _estimator_type = "regressor"
    default_params = {
        "verbosity": -1,
        "force_col_wise": True,
        "num_boost_round": 1000,
        "objective": "rmse",
    }

    def __init__(self, **params):
        """
        Light GBM wrapper
        @param params: Model parameters

        callbacks [list[str]]: Callbacks as a string, transformed in a function here
        """
        self.params = None
        self.set_params(**params)
        self.model = None
        self.callbacks = []
        self.trained = False
        self.num_boost_round = 1000

    @staticmethod
    def convert_to_dataset(x, y=None):
        # Convert input
        assert type(x) in [
            pd.DataFrame,
            pd.Series,
            np.ndarray,
        ], "Unsupported data input format"
        if isinstance(x, np.ndarray) and len(x.shape) == 0:
            x = x.reshape((-1, 1))

        if y is None:
            return lgb.Dataset(x)

        else:
            assert type(y) in [pd.Series, np.ndarray], "Unsupported data label format"
            return lgb.Dataset(x, label=y)

    def fit(self, x, y):
        assert isinstance(x, np.ndarray) or isinstance(x, pd.DataFrame), (
            "X needs to be of type np.ndarray or " "pd.DataFrame"
        )
        assert isinstance(y, np.ndarray) or isinstance(
            y, pd.Series
        ), "Y needs to be of type np.ndarray or pd.Series"
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        # Split & Convert data
        train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size=0.1, shuffle=True
        )
        d_train = self.convert_to_dataset(train_x, train_y)
        d_test = self.convert_to_dataset(test_x, test_y)

        # Get params and split num_boost_round
        params = self.get_params()
        num_boost_round = params.pop("num_boost_round")

        # Model training
        self.model = lgb.train(
            params,
            d_train,
            num_boost_round=num_boost_round,
            valid_sets=[d_train, d_test],
            feval=self.eval_function,
            callbacks=[lgb.early_stopping(100, verbose=False)] + self._get_callbacks(),
        )
        self.trained = True

    def predict(self, x, *args, **kwargs):
        # todo check input data
        assert self.trained is True, "Model not yet trained"
        return self.model.predict(x, *args, **kwargs)

    def set_params(self, **params):
        # Add default
        for k, v in self.default_params.items():
            if k not in params:
                params[k] = v

        # Remove fit options
        if "callbacks" in params:
            self.callbacks = params.pop("callbacks")

        # Update class
        self.num_boost_round = params.pop("num_boost_round")
        self.params = params
        return self

    def _get_callbacks(self) -> list:
        # Output
        callbacks = []

        # Iterate through callbacks
        for callback in self.callbacks:
            # todo implement various callbacks - recognize string and add function
            raise NotImplementedError("Callback not implemented.")

        # Return
        return callbacks

    def get_params(self, **args):
        params = deepcopy(self.params)
        if "deep" in args:
            return params
        if self.callbacks is not None:
            params["callbacks"] = self.callbacks
        params["num_boost_round"] = self.num_boost_round
        return params

    @staticmethod
    def eval_function(prediction, d_train):
        target = d_train.get_label()
        weight = d_train.get_weight()
        # Return mean absolute error
        return (
            "mean_absolute_error",
            mean_absolute_error(target, prediction, sample_weight=weight),
            True,
        )
