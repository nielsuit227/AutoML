import unittest
import numpy as np
import pandas as pd
from Amplo.AutoML import DriftDetector


def draw(n):
    return pd.DataFrame({
        'norm': np.random.normal(0.23, 0.53, n),
        'uniform': np.random.uniform(0, 100, n),
        'exponential': np.random.exponential(0.4, n),
        'gamma': np.random.gamma(0.3, 0.9, n),
        'beta': np.random.beta(0.2, 0.4, n),
    })


class TestDriftDetector(unittest.TestCase):

    def test_distribution_fits(self):
        # Setup
        dists = ["norm", "uniform", "expon", "gamma", "beta"]
        ref = draw(500)
        test = ref.iloc[np.random.permutation(len(ref))[:10]]
        drift = DriftDetector(num_cols=ref.keys())
        drift.fit(ref)

        # Checks
        assert len(drift.check(test)) == 0, "Test data found inconsistent"
        assert len(drift.check(ref.max() + 1)) == len(dists), "Maxima not detected"
        assert len(drift.check(ref.min() - 1)) == len(dists), "Minima not detected"

    def test_categorical(self):
        df = pd.DataFrame({'a': ['a', 'b', 'c', 'd', 'a', 'b', 'a', 'b', 'c', 'a']})
        drift = DriftDetector(cat_cols=['a'])
        drift.fit(df)
        assert 'a' in drift.bins, "Column 'a' rejected."
        assert drift.bins['a'] == {'a': 4, 'b': 3, 'c': 2, 'd': 1}

    def test_add_bins(self):
        df = pd.DataFrame({'a': ['a', 'b', 'c', 'd', 'a', 'b', 'a', 'b', 'c', 'a']})
        drift = DriftDetector(cat_cols=['a'])
        drift.fit(df)
        new_bins = drift.add_bins(drift.bins, df)
        assert new_bins['a'] == {'a': 8, 'b': 6, 'c': 4, 'd': 2}
