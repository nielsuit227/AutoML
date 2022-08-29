#  Copyright (c) 2022 by Amplo.

import os
from typing import Union

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import get_scorer

from amplo.base.exceptions import NotFittedError
from amplo.classification import (
    CatBoostClassifier,
    LGBMClassifier,
    StackingClassifier,
    XGBClassifier,
)
from amplo.regression import (
    CatBoostRegressor,
    LGBMRegressor,
    StackingRegressor,
    XGBRegressor,
)

Model = Union[
    CatBoostClassifier,
    LGBMClassifier,
    XGBClassifier,
    StackingClassifier,
    CatBoostRegressor,
    LGBMRegressor,
    XGBRegressor,
    StackingRegressor,
]


setup_class_params = [
    CatBoostClassifier,
    LGBMClassifier,
    XGBClassifier,
    StackingClassifier,
    CatBoostRegressor,
    LGBMRegressor,
    XGBRegressor,
    StackingRegressor,
]


@pytest.fixture(scope="class", params=setup_class_params)
def setup_class(request):
    model = request.param

    if "Classifier" in model.__name__:
        x, y = make_classification(n_classes=5, n_informative=15)
        is_classification = True
    elif "Regressor" in model.__name__:
        x, y = make_regression(n_informative=15)
        is_classification = False
    else:
        raise ValueError("Invalid model requested")

    if "Stacking" in model.__name__:
        if is_classification:
            model_params = {"DecisionTreeClassifier__max_depth": 10}
        else:
            model_params = {"DecisionTreeRegressor__max_depth": 10}
    else:
        model_params = {"max_depth": 10}

    request.cls._model = model
    request.cls.model_params = model_params
    request.cls.is_classification = is_classification
    request.cls.x = pd.DataFrame(x)
    request.cls.y = pd.Series(y)
    yield


@pytest.mark.usefixtures("setup_class")
class TestModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Initialize model
        self.model = self._model()
        yield

    def test_set_params(self):
        self.model.set_params(**self.model_params)

    def test_get_params(self):
        self.model.get_params()

    def test_fit_pandas(self):
        self.model.fit(self.x, self.y)

    def test_fit_numpy(self):
        self.model.fit(self.x.to_numpy(), self.y.to_numpy())

    def test_is_fitted(self):
        with pytest.raises(NotFittedError):
            self.model.check_is_fitted()
        self.model.fit(self.x, self.y)
        self.model.check_is_fitted()

    def test_predict(self):
        self.model.fit(self.x, self.y)
        prediction = self.model.predict(self.x)

        assert len(prediction.shape) == 1
        if self.is_classification:
            assert np.allclose(prediction.astype("int"), prediction)

    def test_predict_proba(self):
        if self.is_classification:
            self.model.fit(self.x, self.y)
            prediction = self.model.predict_proba(self.x)

            assert not np.isnan(prediction).any(), "NaN in prediction: {}".format(
                prediction
            )
            assert len(prediction.shape) == 2
            assert prediction.shape[1] == len(np.unique(self.y))
            assert np.allclose(np.sum(prediction, axis=1), 1)

    def test_cloneable(self):
        clone_model: Model = clone(self.model.set_params(**self.model_params))  # noqa
        clone_model.fit(self.x, self.y)

        if self.is_classification:
            prediction = clone_model.predict_proba(self.x)
            assert len(prediction.shape) != 1
            assert np.allclose(np.sum(prediction, axis=1), 1)
        else:
            prediction = clone_model.predict(self.x)
            assert len(prediction.shape) == 1

    def test_scorer(self):
        self.model.fit(self.x, self.y)

        if self.is_classification:
            scorers = ["neg_log_loss", "accuracy", "f1_micro"]
        else:
            scorers = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]

        for name in scorers:
            get_scorer(name)(self.model, self.x, self.y)

    def test_binary(self):
        if self.is_classification:
            x, y = make_classification()
            self.model.fit(x, y)
            self.model.predict(x)
            self.model.predict_proba(x)
            get_scorer("neg_log_loss")(self.model, x, y)
            get_scorer("accuracy")(self.model, x, y)
            get_scorer("f1_micro")(self.model, x, y)

    def test_pickleable(self):
        x, y = make_classification()
        self.model.fit(x, y)
        joblib.dump(self.model, "tmp.joblib")
        os.remove("tmp.joblib")
