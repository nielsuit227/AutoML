#  Copyright (c) 2022 by Amplo.

import os

import numpy as np
import pandas as pd
import pytest

from amplo import Pipeline
from tests import get_all_modeller_models


class TestPipeline:
    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_main_predictors(self, mode, make_x_y):
        # Test mode
        x, y = make_x_y
        x = x.iloc[:, :5]  # for speed up
        pipeline = Pipeline(n_grid_searches=0, plot_eda=False, extract_features=False)
        pipeline.fit(x, y)
        x_c, _ = pipeline.convert_data(x)

        models = get_all_modeller_models(mode)
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
        pipeline = Pipeline(no_dirs=True, n_grid_searches=0, extract_features=False)
        pipeline.fit(x, y)
        assert not os.path.exists("Auto_ML"), "Directory created"

    @pytest.mark.parametrize("mode", ["regression"])
    def test_no_args(self, mode, make_x_y):
        x, y = make_x_y
        pipeline = Pipeline(n_grid_searches=0)
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
        pipeline = Pipeline(n_grid_searches=0)
        pipeline.fit(x, y)

        # Test Directories
        assert os.path.exists("Auto_ML")
        assert os.path.exists("Auto_ML/Data")
        assert os.path.exists("Auto_ML/Features")
        assert os.path.exists("Auto_ML/Production")
        assert os.path.exists("Auto_ML/Documentation")
        assert os.path.exists("Auto_ML/Results.csv")

    @pytest.mark.parametrize("mode", ["classification"])
    def test_data_organisation(self, mode, make_x_y):
        x, y = make_x_y
        pipeline = Pipeline(n_grid_searches=0)
        pipeline._read_data(x, y)
        pipeline._mode_detector()
        pipeline._data_processing()
        assert os.path.exists("Auto_ML/Data/Cleaned_v1.csv")
        pipeline._feature_processing()
        assert not os.path.exists("Auto_ML/Data/Cleaned_v1.csv")
        assert os.path.exists("Auto_ML/Data/Extracted_v1.csv")
        pipeline._initial_modelling()
        pipeline.conclude_fitting()
        assert len(os.listdir("Auto_ML/Production/v1/")) > 0
        pipeline = Pipeline(n_grid_searches=0)
        assert pipeline.version == 2
        pipeline._read_data(x, y)
        pipeline._mode_detector()
        pipeline._data_processing()
        assert os.path.exists("Auto_ML/Data/Cleaned_v2.csv")
        assert not os.path.exists("Auto_ML/Data/Extracted_v1.csv")

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
        pipeline = Pipeline(target="TARGET", n_grid_searches=0, extract_features=False)
        pipeline.fit(df)
