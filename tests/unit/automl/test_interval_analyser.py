#  Copyright (c) 2022 by Amplo.

from pathlib import Path

import numpy as np
import pandas as pd

from amplo.automl import IntervalAnalyser
from tests import create_data_frames, rmtree


class TestIntervalAnalyser:
    @classmethod
    def setup_class(cls):
        log_files = [f"log_{i}" for i in range(20)]
        index = pd.MultiIndex.from_product([log_files, range(100)])
        dfs = []
        labels = []
        for i in range(10):
            df1, df2 = create_data_frames(n_samples=100, n_features=5)
            dfs.extend([df1, df2])
            labels.extend([1] * 100)
            labels.extend([0] * 100)
        df = pd.concat(dfs, axis=0)
        df = df.set_index(index, drop=True)
        df = (df - df.min()) / (df.max() - df.min())
        df["target"] = labels
        cls.df = df

    def test_all(self):
        # Normal run
        ia = IntervalAnalyser(min_length=10, target="target")
        data_no_noise = ia.fit_transform(self.df)

        # Attribute tests
        assert ia.n_samples_ == 2000
        assert ia.n_columns_ == 5
        assert ia.n_folders_ == 2
        assert ia.n_files_ == 20

        # Functional tests
        assert ia.distributions_ is not None
        for log in ia.distributions_.index.get_level_values(0):
            dist = np.array(ia.distributions_[log].values)
            assert all(
                v < 0.8 for v in dist[: int(len(dist) / 2)]
            ), "Noise with high percentage of neighbors"
            assert all(
                v > 0.8 for v in dist[int(len(dist) / 2) :]
            ), "Information with low percentage of neighbors"

        # Data tests
        assert (
            len(self.df) == ia.n_samples_ and len(data_no_noise) * 2 == ia.n_samples_
        ), "Incorrect number of samples"
        assert (
            self.df.index.get_level_values(0).nunique() == ia.n_files_
        ), "Files skipped"
        assert len(self.df.keys()) == len(
            data_no_noise.keys()
        ), "Incorrect number of features"

    # TODO test correlation warning
    # TODO test data quality warning
    # TODO test multi-index error
    # TODO test index misalignment error
    # TODO test feature/sample length error
    # TODO test target != labels.name
