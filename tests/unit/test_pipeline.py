import os

import numpy as np
import pandas as pd
import pytest

from Amplo import Pipeline
from Amplo.AutoML import Modeller


class TestPipeline:
    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_main_predictors(self, mode, make_x_y):
        # Test mode
        x, y = make_x_y
        x = x.iloc[:, :5]  # for speed up
        pipeline = Pipeline(
            grid_search_iterations=0, plot_eda=False, extract_features=False
        )
        pipeline.fit(x, y)
        x_c, _ = pipeline.convert_data(x)

        models = {
            *Modeller(mode=mode, samples=100).return_models(),
            *Modeller(mode=mode, samples=100_000).return_models(),
        }
        for model in models:
            model.fit(x_c, y)
            pipeline.best_model = model
            pipeline.predict(x)
            assert isinstance(
                pipeline._main_predictors, dict
            ), "Main predictors not dictionary."

    @pytest.mark.parametrize("mode", ["classification"])
    def test_no_dirs(self, mode, make_x_y):
        x, y = make_x_y
        pipeline = Pipeline(
            no_dirs=True, grid_search_iterations=0, extract_features=False
        )
        pipeline.fit(x, y)
        assert not os.path.exists("AutoML"), "Directory created"

    @pytest.mark.parametrize("mode", ["regression"])
    def test_no_args(self, mode, make_x_y):
        x, y = make_x_y
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline.fit(x, y)

    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_mode_detector(self, mode, make_x_y):
        x, y = make_x_y
        pipeline = Pipeline()
        pipeline._read_data(x, y)._mode_detector()
        assert pipeline.mode == mode

    @pytest.mark.parametrize("mode", ["classification"])
    def test_create_folders(self, mode, make_x_y):
        x, y = make_x_y
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline.fit(x, y)

        # Test Directories
        assert os.path.exists("AutoML")
        assert os.path.exists("AutoML/Data")
        assert os.path.exists("AutoML/Features")
        assert os.path.exists("AutoML/Production")
        assert os.path.exists("AutoML/Documentation")
        assert os.path.exists("AutoML/Results.csv")

    def test_read_write_csv(self):
        """
        Check whether intermediate data is stored and read correctly
        """
        # Set path
        data_path = "test_data.csv"

        # Test single index
        data_write = pd.DataFrame(
            np.random.randint(0, 100, size=(10, 10)),
            columns=[f"feature_{i}" for i in range(10)],
            dtype="int64",
        )
        data_write.index.name = "index"
        Pipeline()._write_csv(data_write, data_path)
        data_read = Pipeline._read_csv(data_path)
        assert data_write.equals(
            data_read
        ), "Read data should be equal to original data"

        # Test multi-index (cf. IntervalAnalyser)
        data_write = data_write.set_index(data_write.columns[-2:].to_list())
        data_write.index.names = ["log", "index"]
        Pipeline()._write_csv(data_write, data_path)
        data_read = Pipeline._read_csv(data_path)
        assert data_write.equals(
            data_read
        ), "Read data should be equal to original data"

        # Remove data
        os.remove(data_path)

    @pytest.mark.parametrize("mode", ["classification"])
    def test_capital_target(self, mode, make_x_y):
        x, y = make_x_y
        df = pd.DataFrame(x)
        df["TARGET"] = y
        pipeline = Pipeline(
            target="TARGET", grid_search_iterations=0, extract_features=False
        )
        pipeline.fit(df)
