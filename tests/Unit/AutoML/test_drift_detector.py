import json
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
        yp = np.random.randint(0, 2, (100))
        drift = DriftDetector(cat_cols=['a'])
        drift.fit(df)

        # Test empty
        assert drift.add_bins({}, df)
        assert drift.add_output_bins((), yp)

        # Test actual adding
        new_bins = drift.add_bins(drift.bins, df)
        assert new_bins['a'] == {'a': 8, 'b': 6, 'c': 4, 'd': 2}

    def test_storable(self):
        df = pd.DataFrame({'a': ['a', 'b', 'c'], 'b': [0, 0.1, 0.2]})
        drift = DriftDetector(cat_cols=['a'], num_cols=['b'])
        drift.fit(df)
        json.dumps(drift.bins)
        json.dumps(drift.add_bins(drift.bins, df))
        pred = np.random.randint(0, 2, (100))
        old = drift.add_output_bins((), pred)
        drift.add_output_bins(old, pred)

