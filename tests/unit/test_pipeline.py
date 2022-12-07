#  Copyright (c) 2022 by Amplo.

import os

import numpy as np
import pandas as pd
import pytest

from amplo import Pipeline
from tests import get_all_modeller_models


class TestPipeline:
    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_main_predictors(self, mode, make_data):
        # Test mode
        data = make_data
        pipeline = Pipeline(n_grid_searches=0, plot_eda=False, extract_features=False)
        pipeline.fit(data)
        x_c = pipeline.transform(data)

        models = get_all_modeller_models(mode)
        for model in models:
            model.fit(x_c, data["target"])
            pipeline.best_model_ = model
            pipeline.predict(data)
            assert isinstance(
                pipeline.main_predictors_, dict
            ), f"Main predictors not dictionary: {type(pipeline.main_predictors_)}"

    @pytest.mark.parametrize("mode", ["classification"])
    def test_no_dirs(self, mode, make_data):
        data = make_data
        pipeline = Pipeline(no_dirs=True, n_grid_searches=0, extract_features=False)
        pipeline.fit(data)
        assert not os.path.exists("Auto_ML"), "Directory created"

    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_mode_detector(self, mode, make_data):
        data = make_data
        pipeline = Pipeline()
        pipeline._mode_detector(data)
        assert pipeline.mode == mode

    @pytest.mark.parametrize("mode", ["classification"])
    def test_create_folders(self, mode, make_data):
        data = make_data
        pipeline = Pipeline(n_grid_searches=0)
        pipeline.fit(data)

        # Test Directories
        assert os.path.exists("Auto_ML")
        assert os.path.exists("Auto_ML/Model.joblib")
        assert os.path.exists("Auto_ML/Settings.json")

    def test_read_write_df(self):
        """
        Check whether intermediate data is stored and read correctly
        """
        # Set path
        data_path = "test_data.parquet"

        # Test single index
        data_write = pd.DataFrame(
            np.random.randint(0, 100, size=(10, 10)),
            columns=[f"feature_{i}" for i in range(10)],
            dtype="int64",
        )
        data_write.index.name = "index"
        Pipeline()._write_df(data_write, data_path)
        data_read = Pipeline._read_df(data_path)
        assert data_write.equals(
            data_read
        ), "Read data should be equal to original data"

        # Test multi-index (cf. IntervalAnalyser)
        data_write = data_write.set_index(data_write.columns[-2:].to_list())
        data_write.index.names = ["log", "index"]
        Pipeline()._write_df(data_write, data_path)
        data_read = Pipeline._read_df(data_path)
        assert data_write.equals(
            data_read
        ), "Read data should be equal to original data"

        # Remove data
        os.remove(data_path)

    @pytest.mark.parametrize("mode", ["classification"])
    def test_capital_target(self, mode, make_data):
        data = make_data
        data["TARGET"] = data["target"]
        data = data.drop("target", axis=1)
        pipeline = Pipeline(target="TARGET", n_grid_searches=0, extract_features=False)
        pipeline.fit(data)
