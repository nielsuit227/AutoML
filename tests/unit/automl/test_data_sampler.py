#  Copyright (c) 2022 by Amplo.

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from amplo.automl import DataSampler


class TestDataSampler:
    def test_binary_small_under(self):
        print("Binary Small Under")
        # Data
        x, y = make_classification(n_classes=2, n_samples=100, weights=[0.05, 0.95])
        x, y = pd.DataFrame(x), pd.Series(y)

        # Resample
        sampler = DataSampler(method="under")
        x_res, y_res = sampler.fit_resample(x, y)

        # Tests
        assert len(y_res) <= len(y), "y_res longer than y"

    def test_binary_small_over(self):
        print("Binary Small Over")
        # Data
        x, y = make_classification(n_classes=2, n_samples=1000, weights=[0.05, 0.95])
        x, y = pd.DataFrame(x), pd.Series(y)

        # Resample
        sampler = DataSampler(method="over")
        x_res, y_res = sampler.fit_resample(x, y)

        # Tests
        assert len(y_res) >= len(y), "y_res shorter than y"

    def test_binary_big_under(self):
        print("Binary Big Under")
        # Data
        x, y = make_classification(n_classes=2, n_samples=100000, weights=[0.05, 0.95])
        x, y = pd.DataFrame(x), pd.Series(y)

        # Resample
        sampler = DataSampler(method="under")
        x_res, y_res = sampler.fit_resample(x, y)

        # Tests
        assert len(y_res) <= len(y), "y_res longer than y"

    def test_binary_big_over(self):
        print("Binary Big Over")
        # Data
        x, y = make_classification(n_classes=2, n_samples=100000, weights=[0.05, 0.95])
        x, y = pd.DataFrame(x), pd.Series(y)

        # Resample
        sampler = DataSampler(method="over")
        x_res, y_res = sampler.fit_resample(x, y)

        # Tests
        assert len(y_res) >= len(y), "y_res shorter than y"
        assert sampler.opt_sampler is None, "Shouldn't over-sample big data"

    def test_multi_small_under(self):
        print("Multi Small Under")
        # Data
        x, y = make_classification(
            n_classes=5,
            n_features=10,
            n_informative=10,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            n_samples=100,
            weights=[0.05, 0.2, 0.1, 0.05, 0.6],
        )
        x, y = pd.DataFrame(x), pd.Series(y)

        # Resample
        sampler = DataSampler(method="under")
        x_res, y_res = sampler.fit_resample(x, y)

        # Tests
        assert len(y_res) <= len(y), "y_res longer than y"
        assert sampler.opt_sampler is None, "Shouldn't under-sample on small data"

    def test_multi_small_over(self):
        print("Multi Small Over")
        # Data
        x, y = make_classification(
            n_classes=5,
            n_features=10,
            n_informative=10,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            n_samples=100,
            weights=[0.05, 0.2, 0.1, 0.05, 0.6],
        )
        x, y = pd.DataFrame(x), pd.Series(y)

        # Resample
        sampler = DataSampler(method="over")
        x_res, y_res = sampler.fit_resample(x, y)

        # Tests
        assert len(y_res) >= len(y), "y_res shorter than y"

    def test_multi_big_under(self):
        print("Multi Big Under")
        # Data
        x, y = make_classification(
            n_classes=5,
            n_features=10,
            n_informative=10,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            n_samples=100000,
            weights=[0.05, 0.2, 0.1, 0.05, 0.6],
        )
        x, y = pd.DataFrame(x), pd.Series(y)

        # Resample
        sampler = DataSampler(method="under")
        x_res, y_res = sampler.fit_resample(x, y)

        # Tests
        assert len(y_res) <= len(y), "y_res longer than y"

    def test_multi_big_over(self):
        print("Multi Big Over")
        # Data
        x, y = make_classification(
            n_classes=5,
            n_features=10,
            n_informative=10,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            n_samples=100000,
            weights=[0.05, 0.2, 0.1, 0.05, 0.6],
        )
        x, y = pd.DataFrame(x), pd.Series(y)

        # Resample
        sampler = DataSampler(method="over")
        x_res, y_res = sampler.fit_resample(x, y)

        # Tests
        assert len(y_res) <= len(y), "y_res longer than y"
        assert sampler.opt_sampler is None, "Shouldn't over-sample on big data"

    def test_balanced(self):
        print("Balanced")
        # Data
        x, y = make_classification(n_samples=100)
        x, y = pd.DataFrame(x), pd.Series(y)

        # Resample
        sampler = DataSampler(method="both")
        x_res, y_res = sampler.fit_resample(x, y)

        assert np.allclose(x_res, x), "X is not the same"
        assert sampler.opt_sampler is None, "Classes are balanced"
