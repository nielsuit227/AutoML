import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from Amplo.AutoML import FeatureProcesser


class TestFeatureProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_regression(self):
        x, y = make_regression()
        x, y = pd.DataFrame(x), pd.Series(y)
        fp = FeatureProcesser(max_lags=2, mode='regression')
        xt, sets = fp.fit_transform(x, y)

    def test_classification(self):
        x, y = make_classification()
        x, y = pd.DataFrame(x), pd.Series(y)
        fp = FeatureProcesser(max_lags=2, mode='classification')
        xt, sets = fp.fit_transform(x, y)

    def test_co_linearity(self):
        y = pd.Series(np.linspace(2, 100, 100))
        x = pd.DataFrame({'a': np.linspace(-4, 4, 100), 'b': np.linspace(-4, 4, 100)})
        fp = FeatureProcesser(mode='regression')
        xt, sets = fp.fit_transform(x, y)
        assert len(fp.coLinearFeatures) != 0, "Colinear feature not removed"

    def test_multiply_features(self):
        y = pd.Series(np.linspace(2, 100, 100))
        b = pd.Series(np.linspace(-4, 4, 100) ** 2)
        x = pd.DataFrame({'a': y / b, 'b': b})
        fp = FeatureProcesser(mode='regression')
        xt, sets = fp.fit_transform(x, y)
        assert len(fp.crossFeatures) != 0, "Multiplicative feature not spotted"

    def test_division(self):
        y = pd.Series(np.linspace(2, 100, 100))
        b = pd.Series(np.linspace(-4, 4, 100) ** 2)
        x = pd.DataFrame({'a': y * b, 'b': b})
        fp = FeatureProcesser(mode='regression')
        xt, sets = fp.fit_transform(x, y)
        assert len(fp.crossFeatures) != 0, "Division feature not spotted"

    def test_trigonometry(self):
        y = pd.Series(np.sin(np.linspace(0, 100, 100)))
        x = pd.DataFrame({'a': np.linspace(0, 100, 100)})
        fp = FeatureProcesser(mode='regression')
        xt, sets = fp.fit_transform(x, y)
        assert len(fp.trigonometricFeatures) != 0, "Trigonometric feature not spotted"

    def test_lagged(self):
        y = pd.Series(np.random.randint(0, 100, 100))
        x = pd.DataFrame({'a': np.roll(y, -5)})
        fp = FeatureProcesser(mode='regression')
        xt, sets = fp.fit_transform(x, y)
        assert len(fp.laggedFeatures) != 0, "Lagged feature not spotted"

    def test_diff(self):
        y = pd.Series(np.random.randint(1, 100, 100))
        x = pd.DataFrame({'a': np.cumsum(y)})
        fp = FeatureProcesser(mode='regression')
        xt, sets = fp.fit_transform(x, y)
        assert len(fp.diffFeatures) != 0, "Difference feature not spotted"

    def test_select(self):
        n = 1000
        y = pd.Series(np.linspace(0, 100, n))
        x = pd.DataFrame({'a': y, 'b': np.random.normal(0, 1, n)})
        fp = FeatureProcesser(mode='regression')
        xt, sets = fp.fit_transform(x, y)
        assert all([len(i) == 1 for i in sets.values()]), f"Random Feature Selected: {sets}"

    def test_settings(self):
        y = pd.Series(np.random.randint(1, 100, 100))
        b = pd.Series(np.linspace(-4, 4, 100))
        x = pd.DataFrame({'a': np.cumsum(y), 'b': np.roll(y, -5), 'c': y / b, 'd': y * b})
        fp = FeatureProcesser(mode='regression')
        xt, sets = fp.fit_transform(x, y)
        settings = fp.get_settings()
        fpn = FeatureProcesser(mode='regression')
        fpn.load_settings(settings)
        for k, v in sets.items():
            xtn = fpn.transform(x, k)
            assert len(v) == len(xtn.keys()), "Incorrect number of keys"
            assert all(xt[v].keys() == xtn.keys()), 'Keys are not correct'
            assert np.allclose(xt[v], xtn), 'Transformed data not consistent for {} set'.format(k)

    def test_get_required(self):
        y = pd.Series(np.linspace(2, 100, 100))
        b = pd.Series(np.linspace(-4, 4, 100) ** 2)
        x = pd.DataFrame({'a': y / b, 'b': b, 'c': b / 2})
        fp = FeatureProcesser(mode='regression')
        xt, sets = fp.fit_transform(x, y)
        assert set(fp.get_required_features(['a__x__b'])) == set(['b', 'a'])
