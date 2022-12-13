#  Copyright (c) 2022 by Amplo.

import json
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss, r2_score

from amplo import Pipeline
from amplo.utils.testing import make_interval_data
from tests import rmtree


@pytest.mark.usefixtures("make_mode")
class TestPipelineByMode:
    target: str
    mode: str
    data: pd.DataFrame

    def test_mode_detector(self):
        # Numerical test
        pipeline = Pipeline(no_dirs=True, target=self.target)
        pipeline._mode_detector(self.data)
        assert pipeline.mode == self.mode

        # Categorical test
        if self.mode == "classification":
            # Convert to categorical
            data = self.data.copy()
            data[self.target] = [f"Class_{v}" for v in data[self.target].values]
            # Detect mode
            pipeline = Pipeline(no_dirs=True, target=self.target)
            pipeline._mode_detector(self.data)
            assert pipeline.mode == self.mode

    def test_dir_and_settings(self):
        pipeline = Pipeline(
            target=self.target,
            mode=self.mode,
            objective="r2" if self.mode == "regression" else "neg_log_loss",
            n_grid_searches=0,
            extract_features=False,
        )
        pipeline.fit(self.data)

        if self.mode == "classification":
            # Pipeline Prediction
            prediction = pipeline.predict_proba(self.data)
            assert log_loss(self.data[self.target], prediction) > -1

        elif self.mode == "regression":
            # Pipeline Prediction
            prediction = pipeline.predict(self.data)
            assert len(prediction.shape) == 1
            assert r2_score(self.data[self.target], prediction) > 0.75

        else:
            raise ValueError(f"Invalid mode {self.mode}")

        # Settings prediction
        settings = json.load(open("Auto_ML/Settings.json", "r"))
        model = joblib.load("Auto_ML/Model.joblib")
        p = Pipeline(no_dirs=True)
        p.load_settings(settings)
        p.load_model(model)
        if self.mode == "classification":
            assert np.allclose(p.predict_proba(self.data), prediction)
        else:
            assert np.allclose(p.predict(self.data), prediction)

        # Test loading function
        p.load()
        if self.mode == "classification":
            assert np.allclose(p.predict_proba(self.data), prediction)
        else:
            assert np.allclose(p.predict(self.data), prediction)

        # Check settings
        assert p.best_model_
        assert p.best_model_str_
        assert p.best_params_
        assert p.best_feature_set_
        assert p.best_score_
        assert "pipeline" in settings
        assert "version" in settings
        assert "model" in settings
        assert "amplo_version" in settings
        assert "params" in settings
        assert "features" in settings
        assert "validation" in settings
        assert "feature_processing" in settings
        assert "data_processing" in settings


class TestPipeline:
    def test_drift(self):
        # todo connect with logger
        pytest.skip("This test is not yet implemented")

    def test_all_models(self):
        # TODO
        pytest.skip("This test is not yet implemented")

    def test_interval_analyzer(self):
        """
        Use interval analyzer when a folder with logs is presented to pipeline.
        """
        # Create dummy data
        data_dir = "./test_data"
        make_interval_data(directory=data_dir, target="target")

        # Pipeline
        pipeline = Pipeline(
            n_grid_searches=0, extract_features=False, interval_analyse=True
        )
        pipeline.fit(data_dir, target="DummyLabel_0")

        # Check if IntervalAnalyser is fitted
        assert (
            pipeline.interval_analyser and pipeline.interval_analyser.is_fitted_
        ), "IntervalAnalyser was not fitted"

        # Remove dummy data
        rmtree(data_dir, must_exist=True)

    def test_with_time(self):
        x = np.vstack(
            (  # type: ignore
                [datetime(2020, 1, 1) + timedelta(days=i) for i in range(12)],
                np.linspace(0, 10, 12),
            )
        ).T
        y = np.linspace(0, 10, 12)
        pipeline = Pipeline(n_grid_searches=0, extract_features=False)
        pipeline.fit(x, y)

        # Do the same, but with temporal
        multi_index = pd.MultiIndex.from_product(
            [[1, 2, 3], [1, 2, 3, 4]], names=["log", "index"]
        )
        df = pd.DataFrame(x, index=multi_index)
        pipeline = Pipeline(n_grid_searches=0)
        pipeline.fit(df, y)

    def test_backwards_compatibility(self):
        # Set up pipeline
        pipeline = Pipeline(main_dir="tests/files/")
        pipeline.load()

        # Predict
        np.random.seed(100)
        df = pd.DataFrame(
            {
                "output_current": np.random.normal(0, 1, size=100),
                "radiator_temp": np.random.normal(0, 1, size=100),
            }
        )
        yp = pipeline.predict_proba(df)
        assert np.allclose(yp, np.array([[0.16389499, 0.83610501]]))
