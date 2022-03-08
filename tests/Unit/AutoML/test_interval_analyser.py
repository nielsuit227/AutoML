import os
import pytest
import shutil
import unittest
import numpy as np
import pandas as pd
from Amplo.AutoML import IntervalAnalyser


def createDataFrames(n_samples, n_features):
    dim = (int(n_samples / 2), n_features)
    columns = [f'Feature_{i}' for i in range(n_features)]
    df1 = pd.DataFrame(
        columns=columns,
        data=np.vstack((np.random.normal(0, 1, dim), np.random.normal(100, 1, dim))))
    df2 = pd.DataFrame(
        columns=columns,
        data=np.vstack((np.random.normal(0, 1, dim), np.random.normal(-100, 1, dim))))
    return df1, df2


class TestIntervalAnalyser(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Settings
        cls.n_samples = 50
        cls.n_features = 25

        # Create classes
        if not os.path.exists('IA/Class_1'):
            os.makedirs('IA/Class_1')
            os.makedirs('IA/Class_2')

        for i in range(140):
            # Create dataframes
            df1, df2 = createDataFrames(cls.n_samples, cls.n_features)
            df1.to_csv(f'IA/Class_1/Log_{i}.csv', index=False)
            df2.to_csv(f'IA/Class_2/Log_{i}.csv', index=False)

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists('IA'):
            shutil.rmtree('IA')
        if os.path.exists('tests/Unit/AutoML/IA'):
            shutil.rmtree('tests/Unit/AutoML/IA')

    def test_all(self):
        ia = IntervalAnalyser(folder='IA', min_length=10)
        df = ia.fit_transform()

        # Attribute tests
        assert ia.n_folders == 2
        assert ia.n_files == 280

        # Functional tests
        for i in range(ia.n_files):
            dist = ia._distributions[i].values
            assert all(v < 0.8 for v in dist[:int(len(dist) / 2)]), 'Noise with high percentage of neighbors'
            assert all(v > 0.8 for v in dist[int(len(dist) / 2):]), 'Information with low percentage of neighbors'
        assert len(df) == int(ia.samples / 2), 'Incorrect number of samples'
        assert df.index.get_level_values(0).nunique() == ia.n_files, 'Files skipped'




