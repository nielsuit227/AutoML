#  Copyright (c) 2022 by Amplo.

from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss, r2_score

import amplo
from amplo import Pipeline
from amplo.base.objects import Result
from tests import create_test_folders, get_all_modeller_models, rmtree


def get_fake_pipe(data: pd.DataFrame, **kwargs) -> Pipeline:
    pipe = Pipeline(extract_features=False, **kwargs)
    pipe._mode_detector(data)
    pipe._set_subclasses()
    assert pipe.feature_processor and pipe.data_processor
    pipe.data_processor.fit_transform(data)
    pipe.feature_processor.fit_transform(data, feature_set="rf")
    pipe.results_ = [
        Result(
            model="CatBoostClassifier",
            params={},
            feature_set="rf_increment",
            score=0.1,
            worst_case=0.1,
            time=0.2,
            date="now",
        )
    ]
    pipe.data_processor.prune_features(pipe.best_features_)
    pipe.is_fitted_ = True
    return pipe


class TestPipeline:
    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_mode_detector(self, mode, data):
        pipe = Pipeline()
        pipe._mode_detector(data)
        assert pipe.mode == mode

    @pytest.mark.parametrize("mode", ["classification"])
    def test_store_best(self, mode, data):
        pipe = get_fake_pipe(data, no_dirs=True)
        pipe.train_val_best(data)
        assert pipe.best_model_ and pipe.best_model_.is_fitted_

    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_main_predictors(self, mode, data):
        pipe = get_fake_pipe(data)

        # Check main predictors
        models = get_all_modeller_models(mode)
        target = data["target"]
        data = pipe.transform(data)
        for model in models:
            model.fit(data, target)
            pipe.best_model_ = model
            pipe.predict(data)
            assert isinstance(
                pipe.main_predictors_, dict
            ), f"Main predictors not dictionary: {type(pipe.main_predictors_)}"

    @pytest.mark.parametrize("mode", ["classification"])
    def test_capital_target(self, mode, data):
        data["TARGET"] = data["target"]
        data = data.drop("target", axis=1)
        pipe = get_fake_pipe(data, target="TARGET")
        assert pipe.target == "TARGET"

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
            Pipeline()._read_data(x, {"a": 1})  # purposely wrong dtype
        with pytest.raises(ValueError):
            # Missing target
            Pipeline()._read_data(data, "label")
        with pytest.raises(ValueError):
            data2 = data.rename(columns={"target": "target2"})
            Pipeline()._read_data(data2)

    def test_backwards_compatibility(self):
        # TODO: Implement as soon as `amplo.dumps` and `amplo.load` is available.
        #  That PR will then (again) radically change the settings structure of dumped
        #  Amplo objects but will bring many improvements with it.
        pytest.skip("Temporarily deprecated...")

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

    @pytest.mark.parametrize("mode", ["classification"])
    def test_dump_load(self, mode, data):
        pipeline = Pipeline(
            target="target",
            mode=mode,
            objective="neg_log_loss",
            n_grid_searches=0,
            extract_features=False,
        )
        with mock.patch(
            "sklearn.model_selection.cross_val_score",
            return_value=[-0.1, -0.1, -0.1, -0.1, -0.1],
        ):
            pipeline.fit(data)

        if mode == "classification":
            # Pipeline Prediction
            prediction = pipeline.predict_proba(data)
            assert log_loss(data["target"], prediction) > -1

        elif mode == "regression":
            # Pipeline Prediction
            prediction = pipeline.predict(data)
            assert len(prediction.shape) == 1
            assert r2_score(data["target"], prediction) > 0.75

        else:
            raise ValueError(f"Invalid mode {mode}")

        # Store and recreate pipeline
        enc = amplo.dumps(pipeline)
        p = amplo.loads(enc)

        assert np.allclose(p.predict_proba(data), prediction)
        assert p.best_model_
        assert p.best_model_str_
        assert p.best_params_
        assert p.best_feature_set_
        assert p.best_score_
