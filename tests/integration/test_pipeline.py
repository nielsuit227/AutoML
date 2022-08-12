#  Copyright (c) 2022 by Amplo.

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss, r2_score

from amplo import Pipeline
from amplo.utils.testing import make_interval_data
from tests import make_data, rmtree


@pytest.fixture(scope="class", params=["regression", "classification"])
def make_mode(request, target="target"):
    mode = request.param
    request.cls.mode = mode
    request.cls.target = target
    request.cls.data = make_data(mode=mode, target=target)
    yield


@pytest.mark.usefixtures("make_mode")
class TestPipelineByMode:
    def test_mode_detector(self):
        # Numerical test
        pipeline = Pipeline(no_dirs=True, target=self.target)
        pipeline._read_data(self.data)._mode_detector()
        assert pipeline.mode == self.mode

        # Categorical test
        if self.mode == "classification":
            # Convert to categorical
            data = self.data.copy()
            data[self.target] = [f"Class_{v}" for v in data[self.target].values]
            # Detect mode
            pipeline = Pipeline(no_dirs=True, target=self.target)
            pipeline._read_data(data)._mode_detector()
            assert pipeline.mode == self.mode

    def test_stacking(self):
        # Set stacking model
        if self.mode == "classification":
            stacking_model = "StackingClassifier"
        else:
            stacking_model = "StackingRegressor"
        # Fit pipeline
        pipeline = Pipeline(
            target=self.target,
            grid_search_type=None,
            stacking=True,
            extract_features=False,
        )
        pipeline.fit(self.data, model=stacking_model)

    def test_dir_and_settings(self):
        pipeline = Pipeline(
            target=self.target,
            mode=self.mode,
            objective="r2" if self.mode == "regression" else "neg_log_loss",
            n_grid_searches=0,
            plot_eda=False,
            process_data=False,
            extract_features=False,
            document_results=False,
        )
        pipeline.fit(self.data)

        # Test Directories
        assert os.path.exists("Auto_ML")
        assert os.path.exists("Auto_ML/Data")
        assert os.path.exists("Auto_ML/Features")
        assert os.path.exists("Auto_ML/Production")
        assert os.path.exists("Auto_ML/Documentation")
        assert os.path.exists("Auto_ML/Results.csv")

        if self.mode == "classification":
            # Pipeline Prediction
            prediction = pipeline.predict_proba(self.data)
            assert log_loss(self.data[self.target], prediction) > -1

        elif self.mode == "regression":
            # Test data handling
            c, _ = pipeline.convert_data(self.data.drop(self.target, axis=1))
            x = pipeline.x[pipeline.settings["features"]]
            y = self.data[self.target]
            assert np.allclose(
                c.values, x.values
            ), f"Inconsistent X: max diff: {np.max(abs(c.values - x.values)):.2e}"
            assert np.allclose(y, pipeline.y), "Inconsistent y-data"

            # Pipeline Prediction
            prediction = pipeline.predict(self.data)
            assert len(prediction.shape) == 1
            assert r2_score(self.data[self.target], prediction) > 0.75

        else:
            raise ValueError(f"Invalid mode {self.mode}")

        # Settings prediction
        settings = json.load(open("Auto_ML/Production/v1/Settings.json", "r"))
        model = joblib.load("Auto_ML/Production/v1/Model.joblib")
        p = Pipeline(no_dirs=True)
        p.load_settings(settings)
        p.load_model(model)
        if self.mode == "classification":
            assert np.allclose(p.predict_proba(self.data), prediction)
        else:
            assert np.allclose(p.predict(self.data), prediction)

        # Check settings
        assert "pipeline" in settings
        assert "version" in settings
        assert "model" in settings
        assert "amplo_version" in settings
        assert "params" in settings
        assert "drift_detector" in settings
        assert "features" in settings
        assert "validation" in settings
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
        make_interval_data(directory=data_dir)

        # Pipeline
        pipeline = Pipeline(n_grid_searches=0)
        pipeline.fit(data_dir)

        # Check if IntervalAnalyser is fitted
        assert pipeline.interval_analyser.is_fitted, "IntervalAnalyser was not fitted"

        # Check data handling
        assert Path(
            "Auto_ML/Data/Interval_Analyzed_v1.csv"
        ).exists(), "Interval-analyzed data was not properly stored"
        assert list(pipeline.data.index.names) == ["log", "index"], "Index is incorrect"
        # TODO: add checks for settings

        # Remove dummy data
        rmtree(data_dir, must_exist=True)

    def test_with_time(self):
        x = np.vstack(
            (
                [datetime(2020, 1, 1) + timedelta(days=i) for i in range(12)],
                np.linspace(0, 10, 12),
            )
        ).T
        y = np.linspace(0, 10, 12)
        pipeline = Pipeline(n_grid_searches=0)
        pipeline.fit(x, y)

        # Do the same, but with temporal
        multi_index = pd.MultiIndex.from_product(
            [[1, 2, 3], [1, 2, 3, 4]], names=["log", "index"]
        )
        df = pd.DataFrame(x, index=multi_index)
        pipeline = Pipeline(n_grid_searches=0)
        pipeline.fit(df, y)
