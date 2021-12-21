import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from Amplo.AutoML import DataProcesser
from Amplo.Utils import check_dataframe_quality


class TestDataProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classification, y = load_iris(return_X_y=True, as_frame=True)
        cls.classification['target'] = y
        x, y = fetch_california_housing(return_X_y=True)
        cls.regression = pd.DataFrame({'target': y})
        for i in range(x.shape[1]):
            cls.regression['feature_{}'.format(i)] = x[:, i]

    def test_regression(self):
        dp = DataProcesser('target')
        cleaned = dp.fit_transform(self.regression)
        assert check_dataframe_quality(cleaned)

    def test_classification(self):
        dp = DataProcesser('target')
        cleaned = dp.fit_transform(self.classification)
        assert check_dataframe_quality(cleaned)

    def test_interpolation(self):
        dp = DataProcesser()
        data = pd.DataFrame({'a': [1, np.nan, np.nan, np.nan, 5], 'b': [1, 2, 3, 4, 5]})
        cleaned = dp.fit_transform(data)
        assert cleaned['a'].tolist() == [1, 0, 0, 0, 5]

    def test_type_detector(self):
        dp = DataProcesser()
        data = pd.DataFrame({
            'a': ['a', 'b', 'c', 'd'],
            'b': [1, 2, 3, 4],
            'c': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']})
        cleaned = dp.fit_transform(data)
        assert pd.api.types.is_integer_dtype(cleaned['b'])
        assert pd.api.types.is_datetime64_any_dtype(cleaned['c'])
        assert 'a_b' in cleaned
        assert 'a_c' in cleaned
        assert 'a_d' in cleaned

    def test_type_detector_with_nan(self):
        dp = DataProcesser()
        data = pd.DataFrame({
            'a': ['a', 'b', np.nan, 'c', 'd'],
            'b': [1, 2, 3, 4, np.nan],
            'c': ['2020-01-01', np.nan, '2020-01-03', '2020-01-04', '2020-01-05']})
        cleaned = dp.fit_transform(data)
        assert pd.api.types.is_integer_dtype(cleaned['b'])
        assert pd.api.types.is_datetime64_any_dtype(cleaned['c'])
        assert 'a_b' in cleaned
        assert 'a_c' in cleaned
        assert 'a_d' in cleaned

    def test_missing_values(self):
        data = pd.DataFrame({
            'a': ['a', 'b', np.nan, 'c', 'd'],
            'b': [1, 2, 3, 4, np.nan],
            'c': ['2020-01-01', np.nan, '2020-01-03', '2020-01-04', '2020-01-05'],
            'd': [1, 2, 3, 4, 5]})

        # Remove rows
        dp = DataProcesser(missing_values='remove_rows')
        cleaned = dp.fit_transform(data)
        assert len(cleaned) == 3, cleaned.head()        # the nan in the categorical is omitted

        # Remove cols
        dp = DataProcesser(missing_values='remove_cols')
        cleaned = dp.fit_transform(data)
        assert len(cleaned.keys()) == 4, cleaned.head()

        # Replace with 0
        dp = DataProcesser(missing_values='zero')
        cleaned = dp.fit_transform(data)
        assert (cleaned.loc[2, ['a_b', 'a_c', 'a_d']] == 0).all()
        assert cleaned['b'][4] == 0
        assert cleaned['c'][1] == 0

        # Interpolate
        dp = DataProcesser(missing_values='interpolate')
        cleaned = dp.fit_transform(data)
        assert cleaned.isna().sum().sum() == 0, cleaned.head()

        # Fill with mean
        dp = DataProcesser(missing_values='mean')
        cleaned = dp.fit_transform(data)
        assert (cleaned.loc[2, ['a_b', 'a_c', 'a_d']] == 0).all()
        assert cleaned['b'][4] == 2.5
        assert cleaned['c'][1] == pd.to_datetime('2020-01-03 6:00:00', utc=True)

    def test_classification_target(self):
        data = pd.DataFrame({'a': [2, 2, 1, 1, 2], 'b': ['class1', 'class2', 'class1', 'class2', 'class1']})

        # Numerical not starting at 0
        dp = DataProcesser(target='a')
        cleaned = dp.fit_transform(data)
        assert set(cleaned['a']) == {0, 1}

        # Categorical
        dp = DataProcesser(target='b')
        cleaned = dp.fit_transform(data)
        assert set(cleaned['b']) == {0, 1}

    def test_outliers(self):
        x = pd.DataFrame({'a': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1e15],
                          'target': np.linspace(0, 1, 24).tolist()})

        # Clip
        dp = DataProcesser(outlier_removal='clip', target='target')
        xt = dp.fit_transform(x)
        assert xt.max().max() < 1e15, "Outlier not removed"
        assert not xt.isna().any().any(), "NaN found"

        # z-score
        dp = DataProcesser(outlier_removal='z-score', target='target')
        xt = dp.fit_transform(x)
        assert xt.max().max() < 1e15, "Outlier not removed"
        assert not xt.isna().any().any(), "NaN found"
        assert np.isclose(dp.transform(pd.DataFrame({'a': [1e14], 'b': [1]})).max().max(), 1e14)
        assert dp.transform(pd.DataFrame({'a': [1e16], 'b': [1]})).max().max() == 1

        # Quantiles
        dp = DataProcesser(outlier_removal='quantiles', target='target')
        xt = dp.fit_transform(x)
        assert xt.max().max() < 1e15, "Outlier not removed"
        assert not xt.isna().any().any(), "NaN found"
        assert dp.transform(pd.DataFrame({'a': [2], 'b': [-2]})).max().max() == 0

    def test_duplicates(self):
        x = pd.DataFrame({'a': [1, 2, 1], 'a': [1, 2, 1], 'b': [3, 1, 3]})
        dp = DataProcesser()
        xt = dp.fit_transform(x)
        assert len(xt) == 2, "Didn't remove duplicate rows"
        assert len(xt.keys()) == 2, "Didn't remove duplicate columns"

    def test_constants(self):
        x = pd.DataFrame({'a': [1, 1, 1, 1, 1], 'b': [1, 2, 3, 5, 6]})
        dp = DataProcesser()
        xt = dp.fit_transform(x)
        assert 'a' not in xt.keys(), "Didn't remove constant column"

    def test_dummies(self):
        x = pd.DataFrame({'a': ['a', 'b', 'c', 'b', 'c', 'a']})
        dp = DataProcesser(cat_cols=['a'])
        xt = dp.fit_transform(x)
        assert 'a' not in xt.keys(), "'a' still in keys"
        assert 'a_b' in xt.keys(), "a_b missing"
        assert 'a_c' in xt.keys(), "a_c missing"
        xt2 = dp.transform(pd.DataFrame({'a': ['a', 'c']}))
        assert np.allclose(xt2.values, pd.DataFrame({'a_b': [0, 0], 'a_c': [0, 1]}).values), "Converted not correct"

    def test_settings(self):
        x = pd.DataFrame({'a': ['a', 'b', 'c', 'b', 'c', 'a'], 'b': [1, 1, 1, 1, 1, 1]})
        dp = DataProcesser(cat_cols=['a'])
        xt = dp.fit_transform(x)
        assert len(xt.keys()) == 2
        settings = dp.get_settings()
        dp2 = DataProcesser()
        dp2.load_settings(settings)
        xt2 = dp2.transform(pd.DataFrame({'a': ['a', 'b'], 'b': [1, 2]}))
        assert np.allclose(pd.DataFrame({'b': [1.0, 2.0], 'a_b': [0, 1], 'a_c': [0, 0]}).values, xt2.values)

    def test_pruner(self):
        x = pd.DataFrame({'a': ['a', 'b', 'c', 'b', 'c', 'a'], 'b': [1, 1, 1, 1, 1, 1]})
        dp = DataProcesser()
        dp.fit_transform(x)
        dp.prune_features(['b'])
        assert dp.int_cols == ['b']
        assert dp.cat_cols == []
