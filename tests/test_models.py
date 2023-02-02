#  Copyright (c) 2022 by Amplo.

import os
from typing import Any, Union
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import get_scorer

from amplo.base.exceptions import NotFittedError
from amplo.classification import CatBoostClassifier, LGBMClassifier, XGBClassifier
from amplo.regression import CatBoostRegressor, LGBMRegressor, XGBRegressor

Model = Union[
    CatBoostClassifier,
    LGBMClassifier,
    XGBClassifier,
    CatBoostRegressor,
    LGBMRegressor,
    XGBRegressor,
]


setup_class_params = [
    CatBoostClassifier,
    LGBMClassifier,
    XGBClassifier,
    CatBoostRegressor,
    LGBMRegressor,
    XGBRegressor,
]


@pytest.fixture(scope="class", params=setup_class_params)
def setup_class(request):
    model = request.param

    if "Classifier" in model.__name__:
        x, y = make_classification(
            n_classes=3,
            n_informative=2,
            n_features=3,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
        )
        is_classification = True
    elif "Regressor" in model.__name__:
        x, y = make_regression(n_informative=2, n_features=3)
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
    model: Model
    _model: type[Model]
    model_params: dict[str, Any]
    is_classification: bool
    x: pd.DataFrame
    y: pd.Series

    @pytest.fixture(autouse=True)
    def setup(self):
        # Initialize model
        self.model = self._model()
        yield

    def test_is_fitted(self):
        # Test is_fitted
        with pytest.raises(NotFittedError):
            self.model.predict(self.x)
        self.model.fit(self.x, self.y)
        prediction = self.model.predict(self.x)

        # Test predict
        assert len(prediction.shape) == 1
        if self.is_classification:
            assert np.allclose(prediction.astype("int"), prediction)

        # Test predict_proba
        if self.is_classification:
            self.model.fit(self.x, self.y)
            prediction = self.model.predict_proba(self.x)

            assert not np.isnan(prediction).any(), "NaN in prediction: {}".format(
                prediction
            )
            assert len(prediction.shape) == 2
            assert prediction.shape[1] == len(np.unique(self.y))
            assert np.allclose(np.sum(prediction, axis=1), 1)

        # test_scorer
        if self.is_classification:
            scorers = ["neg_log_loss", "accuracy", "f1_micro"]
        else:
            scorers = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
        for name in scorers:
            get_scorer(name)(self.model, self.x, self.y)

    def test_set_params(self):
        self.model.set_params(**self.model_params)

    def test_get_params(self):
        self.model.get_params()

    def test_fit_numpy(self):
        with mock.patch(
            f"{self.model.model.__module__}."
            f"{self.model.model.__class__.__name__}.fit"
        ):
            self.model.fit(self.x.to_numpy(), self.y.to_numpy())

    def test_pickleable(self):
        x, y = make_classification()
        self.model.fit(x, y)
        joblib.dump(self.model, "tmp.joblib")
        os.remove("tmp.joblib")

    def test_clonable(self):
        cloned = clone(self.model.set_params(**self.model_params))
        assert cloned.get_params() == self.model.get_params()
