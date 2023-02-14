#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from amplo.classification import CatBoostClassifier
from amplo.observation._base import ProductionWarning
from amplo.observation.model import ModelObserver
from amplo.regression import CatBoostRegressor
from tests import (
    RandomPredictor,
    _OverfitClassifier,
    _OverfitRegressor,
    _RandomClassifier,
    _RandomRegressor,
)


@pytest.fixture
def make_one_to_one_data(mode):
    size = 100
    if mode == "classification":
        linear_col = np.random.choice([0, 1, 2], size)
    elif mode == "regression":
        linear_col = np.random.uniform(0.0, 1.0, size)
    else:
        raise ValueError("Invalid mode")
    yield pd.DataFrame({"x": linear_col, "target": linear_col})


class DelayedRandomPredictor(RandomPredictor):
    def __init__(self, delay: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = delay

    def predict(self, x):
        time.sleep(self.delay)
        return super().predict(x)

    def predict_proba(self, x):
        time.sleep(self.delay)
        return super().predict_proba(x)


class TestModelObserver:
    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_check_model_size(self, mode):
        with warnings.catch_warnings(record=True) as record:
            ModelObserver().check_model_size(RandomPredictor(mode=mode))
        assert not any(
            isinstance(r.message, ProductionWarning) for r in record
        ), "An unnecessary warning was raised while checking model size."

    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_check_better_than_linear(self, mode, make_one_to_one_data):
        data: pd.DataFrame = make_one_to_one_data
        model: _RandomClassifier | _RandomRegressor
        if mode == "classification":
            model = _RandomClassifier()
        else:
            model = _RandomRegressor()

        # Observe
        with pytest.warns(ProductionWarning):
            ModelObserver().check_better_than_linear(
                model=model, data=data, target="target", mode=mode
            )

    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_check_noise_invariance(self, mode, data):
        model: _OverfitClassifier | _OverfitRegressor
        if mode == "classification":
            model = _OverfitClassifier()
        if mode == "regression":
            model = _OverfitRegressor()

        data = pd.concat([data, data], axis=0)
        with pytest.warns(ProductionWarning):
            ModelObserver().check_noise_invariance(
                model, data=data, target="target", mode=mode
            )

        # Should not trigger normally
        ModelObserver().check_noise_invariance(
            model, data=data, target="target", mode=mode
        )

    def test_check_slice_invariance(self):
        """
        This is a complex test. Slice invariance will be triggered with a linear
        model, when 90% of the data is linearly separable, but 10% is displaced
        compared to the fit.

        We just do this for classification for ease, the observer runs
        irrespective of mode.
        """
        # Classification dataset
        data = pd.DataFrame(
            {
                "x": np.concatenate((np.linspace(0, 90, 96), np.ones(4) * 100)),
                "target": np.concatenate(
                    (
                        np.zeros(48),
                        np.ones(48),
                        np.zeros(4),
                    )
                ),
            }
        )

        # Model
        model = LogisticRegression()
        model.fit(data["x"].values.reshape((-1, 1)), data["target"])

        # Observe
        with pytest.warns(ProductionWarning):
            ModelObserver().check_slice_invariance(
                model=model,
                data=data,
                target="target",
                mode="classification",
            )

        # Should not trigger normally
        ModelObserver().check_slice_invariance(
            model=_RandomClassifier(),
            data=data,
            target="target",
            mode="classification",
        )

    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_check_boosting_overfit(self, mode, data):
        n_estimators = 1000
        boost_kwargs = dict(
            n_estimators=n_estimators,
            l2_leaf_reg=0,
            early_stopping_rounds=n_estimators,
            use_best_model=False,
        )
        _model: CatBoostClassifier | CatBoostRegressor
        if mode == "classification":
            _model = CatBoostClassifier(**boost_kwargs)
        else:
            _model = CatBoostRegressor(**boost_kwargs)
        _model.fit(data.drop("target", axis=1), data["target"])

        # Observer
        with pytest.warns(ProductionWarning):
            ModelObserver().check_boosting_overfit(
                _model, data=data, target="target", mode=mode
            )
