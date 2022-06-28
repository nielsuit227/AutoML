import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from Amplo import Pipeline
from Amplo.Classifiers import CatBoostClassifier
from Amplo.Observation._model_observer import ModelObserver
from Amplo.Observation.base import ProductionWarning
from Amplo.Regressors import CatBoostRegressor
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


class TestModelObserver:
    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_better_than_linear(self, mode, make_one_to_one_data):
        x, y = make_one_to_one_data

        # Make pipeline and simulate fit
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline._read_data(x, y)
        pipeline._mode_detector()
        pipeline.best_model = RandomPredictor(mode=mode)

        # Observe
        obs = ModelObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning):
            obs.check_better_than_linear()

    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_noise_invariance(self, mode, make_one_to_one_data):
        x, y = make_one_to_one_data

        # Make pipeline and simulate fit
        pipeline = Pipeline(grid_search_iterations=0)
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

    def test_slice_invariance(self):
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
        pipeline = Pipeline(grid_search_iterations=0, document_results=False)
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
    def test_boosting_overfit(self, mode):
        if mode == "classification":
            x, y = make_classification(class_sep=0.2, n_samples=500, flip_y=0.2)
        else:
            x, y = make_regression(noise=0.6, n_samples=500)

        # Make pipeline and simulate fit
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline._read_data(x, y)
        pipeline._mode_detector()
        if mode == "classification":
            pipeline.best_model = CatBoostClassifier(
                n_estimators=15000, early_stopping_rounds=15000, use_best_model=False,
            )
        else:
            pipeline.best_model = CatBoostRegressor(
                n_estimators=15000, early_stopping_rounds=15000, use_best_model=False,
            )

        # Observer
        obs = ModelObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning):
            obs.check_boosting_overfit()
