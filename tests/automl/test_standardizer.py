#  Copyright (c) 2022 by Amplo.

import numpy as np
import pandas as pd

from amplo.automl.standardization import Standardizer


class TestStandardizer:
    def test_fit_transform(self):
        df = pd.DataFrame(
            {"x1": np.random.normal(10, 1, 1000), "x2": np.random.normal(-10, 15, 1000)}
        )
        s = Standardizer()
        df_standard = s.fit_transform(df)
        assert df_standard.mean().abs().max() < 1e3
        assert df_standard.std().max() < 1.001 and df_standard.std().min() > 0.999
        assert s.cols_ is not None and set(s.cols_) == {"x1", "x2"}
        assert s.means_ is not None and set(s.means_.keys()) == {"x1", "x2"}
        assert s.stds_ is not None and set(s.stds_.keys()) == {"x1", "x2"}

    def test_target_classification(self):
        df = pd.DataFrame(
            {
                "x1": np.random.normal(10, 1, 100),
                "x2": np.random.normal(-10, 15, 100),
                "y": np.random.random_integers(0, 1, 100),
            }
        )
        s = Standardizer(target="y")
        df_standard = s.fit_transform(df)
        assert set(df_standard["y"].unique()) == {0, 1}

    def test_reverse(self):
        df = pd.DataFrame({"x1": np.random.normal(10, 1, 100)})
        s = Standardizer()
        df_standard = s.fit_transform(df)
        reverted = s.reverse(df_standard["x1"].to_numpy().reshape((-1)), "x1")
        assert np.allclose(reverted, df["x1"])
