#  Copyright (c) 2022 by Amplo.

import time
import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from amplo import Pipeline
from amplo.classification import CatBoostClassifier
from amplo.observation._base import ProductionWarning
from amplo.observation.model import ModelObserver
from amplo.regression import CatBoostRegressor
from tests import OverfitPredictor, RandomPredictor


@pytest.fixture
def make_one_to_one_data(mode):
    size = 100
    if mode == "classification":
        linear_col = np.random.choice([0, 1, 2], size)
    elif mode == "regression":
        linear_col = np.random.uniform(0.0, 1.0, size)
    else:
        raise ValueError("Invalid mode")
    x = linear_col.reshape(-1, 1)
    y = linear_col.reshape(-1)
    yield x, y


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
    def test_check_model_size(self, mode, make_x_y):
        pipeline = Pipeline(n_grid_searches=0)
        pipeline._read_data(*make_x_y)
        pipeline._mode_detector()
        pipeline.best_model = RandomPredictor(mode=mode)

        # Observe
        obs = ModelObserver(pipeline=pipeline)
        with warnings.catch_warnings(record=True) as record:
            obs.check_model_size()
        assert not any(
            isinstance(r.message, ProductionWarning) for r in record
        ), "An unnecessary warning was raised while checking model size."

        # Add big chunk of data to model to make it big
        pipeline.best_model.some_data = np.random.normal(size=3_000_000)
        with pytest.warns(ProductionWarning):
            obs.check_model_size()

    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_check_better_than_linear(self, mode, make_one_to_one_data):
        x, y = make_one_to_one_data

        # Make pipeline and simulate fit
        pipeline = Pipeline(n_grid_searches=0)
        pipeline._read_data(x, y)
        pipeline._mode_detector()
        pipeline.best_model = RandomPredictor(mode=mode)

        # Observe
        obs = ModelObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning):
            obs.check_better_than_linear()

    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_check_noise_invariance(self, mode, make_one_to_one_data):
        x, y = make_one_to_one_data

        # Make pipeline and simulate fit
        pipeline = Pipeline(n_grid_searches=0)
        pipeline._read_data(x, y)
        pipeline._mode_detector()
        pipeline.best_model = OverfitPredictor(mode=mode)

        # Observe
        obs = ModelObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning):
            obs.check_noise_invariance()

        # Should not trigger normally
        pipeline.best_model = RandomPredictor(mode=mode)
        obs = ModelObserver(pipeline=pipeline)
        obs.check_noise_invariance()

    def test_check_slice_invariance(self):
        """
        This is a complex test. Slice invariance will be triggered with a linear
        model, when 90% of the data is linearly separable, but 10% is displaced
        compared to the fit.

        We just do this for classification for ease, the observer runs
        irrespective of mode.
        """
        # Classification dataset
        x = np.linspace(0, 10, 100)
        y = np.concatenate(
            (
                np.zeros(48),
                np.ones(48),
                np.zeros(4),
            )
        )

        # Make pipeline and fit
        pipeline = Pipeline(n_grid_searches=0)
        pipeline._read_data(x, y)
        pipeline._mode_detector()
        pipeline._data_processing()
        pipeline._feature_processing()
        pipeline.conclude_fitting(
            model="LogisticRegression", params={}, feature_set="rf_increment"
        )

        # Observe
        obs = ModelObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning):
            obs.check_slice_invariance()

        # Should not trigger normally
        pipeline.best_model = RandomPredictor(mode="classification")
        obs = ModelObserver(pipeline=pipeline)
        obs.check_slice_invariance()

    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_check_boosting_overfit(self, mode):
        if mode == "classification":
            x, y = make_classification(class_sep=0.5, n_samples=100)
        else:
            x, y = make_regression(noise=0.6, n_samples=100)

        # Make pipeline and simulate fit
        pipeline = Pipeline(n_grid_searches=0, cv_splits=2)

        pipeline._read_data(x, y)
        pipeline._mode_detector()
        n_estimators = 1000
        boost_kwargs = dict(
            n_estimators=n_estimators,
            l2_leaf_reg=0,
            early_stopping_rounds=n_estimators,
            use_best_model=False,
        )
        if mode == "classification":
            pipeline.best_model = CatBoostClassifier(**boost_kwargs)
        else:
            pipeline.best_model = CatBoostRegressor(**boost_kwargs)
        pipeline.best_model.fit(x, y)

        # Observer
        obs = ModelObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning):
            obs.check_boosting_overfit()
