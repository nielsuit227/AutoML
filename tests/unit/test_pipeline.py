#  Copyright (c) 2022 by Amplo.

import os

import numpy as np
import pandas as pd
import pytest

from amplo import Pipeline
from tests import create_test_folders, get_all_modeller_models, rmtree


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

    @pytest.mark.parametrize("mode", ["classification"])
    def test_capital_target(self, mode, make_data):
        data = make_data
        data["TARGET"] = data["target"]
        data = data.drop("target", axis=1)
        pipeline = Pipeline(target="TARGET", n_grid_searches=0, extract_features=False)
        pipeline.fit(data)

    def test_read_dir(self):
        create_test_folders("data", n_samples=100, n_features=5)

        # First multi-class
        df1 = Pipeline(target="Class_1")._read_data("data")
        assert "class_1" in df1
        assert len(df1.keys()) == 6
        assert isinstance(df1.index, pd.MultiIndex)
        assert df1.index.get_level_values(0).dtype == "object"

        # Second binary class
        df2 = Pipeline()._read_data("data", "Class_1")
        assert (df1.index == df2.index).all()
        assert "class_1" in df2
        assert set(df2["class_1"].values) == {0, 1}

        # Trigger errors
        with pytest.raises(ValueError):
            Pipeline()._read_data("data", "NoClass")
        with pytest.raises(ValueError):
            Pipeline()._read_data("data")
        with pytest.raises(ValueError):
            Pipeline()._read_data("NoFolder")

        # Cleanup
        rmtree("data")

    def test_read_numpy(self):
        x = np.random.normal(0, 1, (100, 10))
        y = np.random.normal(0, 1, 100)

        # Normal
        data = Pipeline(target="tArGeT")._read_data(x, y)
        assert "target" in data
        assert len(data.keys()) == 11

        # With series
        data = Pipeline()._read_data(x, pd.Series(y, name="label"))
        assert "label" in data
        assert len(data.keys()) == 11

        # Trigger errors
        with pytest.raises(NotImplementedError):
            Pipeline()._read_data(x, None)
        with pytest.raises(ValueError):
            Pipeline()._read_data(x, np.random.normal(0, 1, 101))

    def test_read_pandas(self):
        index = pd.Index(np.linspace(101, 200, 100))
        x = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, 100),
                "x2": np.random.normal(0, 1, 100),
            },
            index=index,
        )
        y = pd.Series(np.random.normal(0, 1, 100), name="target", index=index)
        data = x.copy()
        data["tArGeT"] = y

        # Normal
        d1 = Pipeline()._read_data(x, y)
        d2 = Pipeline()._read_data(data, "tArGeT")
        d3 = Pipeline(target="tArGeT")._read_data(data)
        assert d1.equals(d2) and d1.equals(d3)
        assert "target" in d1
        assert len(d1.keys()) == 3

        # Trigger errors
        with pytest.raises(ValueError):
            # y too long
            Pipeline()._read_data(x, pd.Series(np.random.normal(0, 1, 101)))
        with pytest.warns(Warning):
            # different index
            Pipeline()._read_data(x, y.reset_index(drop=True))
        with pytest.warns(Warning):
            # different name
            Pipeline()._read_data(
                x, pd.Series(np.random.normal(0, 1, 100), index=index, name="label2")
            )
        with pytest.raises(NotImplementedError):
            # wrong type
            Pipeline()._read_data(x, {"a": 1})  # type: ignore
        with pytest.raises(ValueError):
            # Missing target
            Pipeline()._read_data(data, "label")
        with pytest.raises(ValueError):
            data2 = data.rename(columns={"target": "target2"})
            Pipeline()._read_data(data2)
