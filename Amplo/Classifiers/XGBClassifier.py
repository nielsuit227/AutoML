from copy import deepcopy

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


class XGBClassifier:
    _estimator_type = "classifier"
    default_params = {"verbosity": 0, "num_boost_round": 1000}
    has_predict_proba = True

    def __init__(self, **params):
        """
        XG Boost wrapper
        @param params: Model parameters
        """
        self.num_boost_round = 1000
        self.params = None
        self.set_params(**params)
        self.classes_ = None
        self.model = None
        self.callbacks = None
        self.trained = False
        self.binary = False

    @staticmethod
    def convert_to_d_matrix(X, y=None):
        # Convert input
        assert type(X) in [
            pd.DataFrame,
            pd.Series,
            np.ndarray,
        ], "Unsupported data input format"
        if isinstance(X, np.ndarray) and len(X.shape) == 0:
            X = X.reshape((-1, 1))

        if y is None:
            return xgb.DMatrix(X)

        else:
            assert type(y) in [pd.Series, np.ndarray], "Unsupported data label format"
            return xgb.DMatrix(X, label=y)

    def fit(self, X, y):
        # Split & Convert data
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, stratify=y, test_size=0.1
        )
        d_train = self.convert_to_d_matrix(train_x, train_y)
        d_test = self.convert_to_d_matrix(test_x, test_y)

        # Set loss function
        self.classes_ = np.unique(y)
        self.binary = len(self.classes_) == 2
        if self.binary:
            self.params["objective"] = "binary:logistic"
        else:
            self.params["objective"] = "multi:softprob"
            self.params["num_class"] = len(self.classes_)

        # Model training
        self.model = xgb.train(
            self.params,
            d_train,
            evals=[(d_test, "validation"), (d_train, "train")],
            verbose_eval=False,
            num_boost_round=self.num_boost_round,
            callbacks=[self.callbacks] if self.callbacks is not None else None,
            early_stopping_rounds=100,
        )
        self.trained = True

    def predict(self, X, *args, **kwargs):
        # todo check input data
        assert self.trained is True, "Model not yet trained"
        d_predict = self.convert_to_d_matrix(X)
        prediction = self.model.predict(d_predict, *args, **kwargs)

        # Parse into most-likely class
        if self.binary:
            return np.round(prediction)
        else:
            return np.argmax(prediction, axis=1)

    def predict_proba(self, X, *args, **kwargs):
        # todo check input data
        assert self.trained is True, "Model not yet trained"
        d_predict = self.convert_to_d_matrix(X)
        prediction = self.model.predict(d_predict, *args, **kwargs)

        # Parse into probabilities
        if self.binary:
            return np.hstack((1 - prediction, prediction)).reshape((-1, 2), order="F")
        else:
            return prediction

    def set_params(self, **params):
        # Add default
        for k, v in self.default_params.items():
            if k not in params:
                params[k] = v

        # Remove fit args
        if "callbacks" in params:
            self.callbacks = params.pop("callbacks")

        # Update class
        self.num_boost_round = params.pop("num_boost_round")
        self.params = params
        return self

    def get_params(self, **args):
        params = deepcopy(self.params)
        if "deep" in args:
            return params
        if self.callbacks is not None:
            params["callbacks"] = self.callbacks
        params["num_boost_round"] = self.num_boost_round
        return params
