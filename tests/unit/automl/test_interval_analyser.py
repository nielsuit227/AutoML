#  Copyright (c) 2022 by Amplo.

from pathlib import Path

import numpy as np
import pandas as pd

from amplo.automl import IntervalAnalyser
from amplo.utils.testing import make_interval_data
from tests import rmtree


def create_data_frames(n_samples, n_features):
    dim = (int(n_samples / 2), n_features)
    columns = [f"Feature_{i}" for i in range(n_features)]
    df1 = pd.DataFrame(
        columns=columns,
        data=np.vstack((np.random.normal(0, 1, dim), np.random.normal(100, 1, dim))),
    )
    df2 = pd.DataFrame(
        columns=columns,
        data=np.vstack((np.random.normal(0, 1, dim), np.random.normal(-100, 1, dim))),
    )
    return df1, df2


def create_test_folders(directory: Path, n_samples, n_features):
    # Make directories
    for sub_folder in ("Class_1", "Class_2"):
        (directory / sub_folder).mkdir(exist_ok=True, parents=True)

    # Create and save dataframes
    for i in range(140):
        df1, df2 = create_data_frames(n_samples, n_features)
        df1.to_csv(directory / f"Class_1/Log_{i}.csv", index=False)
        df2.to_csv(directory / f"Class_2/Log_{i}.csv", index=False)


class TestIntervalAnalyser:
    @classmethod
    def setup_class(cls):
        cls.n_samples = 50
        cls.n_features = 25
        cls.ia_folder = Path("IA")
        rmtree(cls.ia_folder)
        create_test_folders(cls.ia_folder, cls.n_samples, cls.n_features)

    @classmethod
    def teardown_class(cls):
        rmtree(cls.ia_folder)

    def test_all(self):
        # Set up IntervalAnalyser
        ia = IntervalAnalyser(min_length=10)
        ia.fit_transform(self.ia_folder)

        # Attribute tests
        assert ia.n_folders == 2
        assert ia.n_files == 280

        # Functional tests
        for i in range(ia.n_files):
            dist = ia._distributions[i].values
            assert all(
                v < 0.8 for v in dist[: int(len(dist) / 2)]
            ), "Noise with high percentage of neighbors"
            assert all(
                v > 0.8 for v in dist[int(len(dist) / 2) :]
            ), "Information with low percentage of neighbors"

        # Data tests
        data_no_noise = ia.data_without_noise
        data_full = ia.data_with_noise
        assert (
            len(data_full) == ia.n_samples and len(data_no_noise) == ia.n_samples // 2
        ), "Incorrect number of samples"
        assert (
            data_full.index.get_level_values(0).nunique() == ia.n_files
        ), "Files skipped"
        assert (
            len(ia._features.columns) <= self.n_features
        ), "Incorrect number of features"

    def test_parse_data(self):
        # Test receiving dataframe and receiving directory
        for test in ("dataframe", "single_dataframe", "directory"):
            ia = IntervalAnalyser(target="labels")

            if test == "dataframe":
                # Create dummy data
                features = make_interval_data(
                    directory=None, target=ia.target, cat_choices=False
                )
                labels = features.pop(ia.target)
                # Set parse arguments
                parse_args = (features, labels)
            elif test == "single_dataframe":
                # Create dummy data
                data = make_interval_data(
                    directory=None, target=ia.target, cat_choices=False
                )
                # Set parse arguments
                parse_args = (data,)
            elif test == "directory":
                # Create dummy data
                make_interval_data(
                    directory=self.ia_folder, target=ia.target, cat_choices=False
                )
                # Set parse arguments
                parse_args = (self.ia_folder, None)
            else:
                raise NotImplementedError()

            # Read data
            _, _ = ia._parse_data(*parse_args)

            # Data checks
            assert all(
                isinstance(data, (pd.Series, pd.DataFrame))
                for data in (ia._labels, ia._features, ia.orig_data)
            ), "Data should not be empty"
            assert (
                len(ia._features.select_dtypes(include=["datetime64"]).columns) == 0
            ), "Date-time columns should have been eliminated, internally"
            assert (
                len(ia._features.select_dtypes(include=["float64"]).columns) == 0
            ), "Dtype float64 columns should have been converted to float32"

            # Other checks
            assert not ia.is_fitted, "IntervalAnalyser should not yet be fitted"
            assert ia._noise_indices is None, "No noise indices should be set yet"

    # TODO test correlation warning
    # TODO test data quality warning
    # TODO test multi-index error
    # TODO test index misalignment error
    # TODO test feature/sample length error
    # TODO test target in features error
    # TODO test target != labels.name
