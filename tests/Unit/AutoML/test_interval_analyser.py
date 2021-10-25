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

            if i <= 20:
                df1.to_csv(f'IA/Class_1/Log_{i}.csv', index=False)
                df2.to_csv(f'IA/Class_2/Log_{i}.csv', index=False)
            elif i <= 40:
                df1.to_json(f'IA/Class_1/Log_{i}.json')
                df2.to_json(f'IA/Class_2/Log_{i}.json')
            elif i <= 60:
                df1.to_xml(f'IA/Class_1/Log_{i}.xml', index=False)
                df2.to_xml(f'IA/Class_2/Log_{i}.xml', index=False)
            elif i <= 80:
                df1.to_feather(f'IA/Class_1/Log_{i}.feather')
                df2.to_feather(f'IA/Class_2/Log_{i}.feather')
            elif i <= 100:
                df1.to_parquet(f'IA/Class_1/Log_{i}.parquet', index=False)
                df2.to_parquet(f'IA/Class_2/Log_{i}.parquet', index=False)
            elif i <= 120:
                df1.to_stata(f'IA/Class_1/Log_{i}.stata', write_index=False)
                df2.to_stata(f'IA/Class_2/Log_{i}.stata', write_index=False)
            elif i <= 140:
                df1.to_pickle(f'IA/Class_1/Log_{i}.pickle')
                df2.to_pickle(f'IA/Class_2/Log_{i}.pickle')
                
    def test_all(self):
        int_an = IntervalAnalyser(folder='IA')
        sens = int_an.analyse()

        assert len(sens) == self.n_samples
        assert sens.ndim == 1
        assert sens[-1] > sens[0]
        assert np.max(sens) <= 100
        assert np.min(sens) >= 0


@pytest.fixture(autouse=True)
def teardown():
    yield
    if os.path.exists('IA'):
        shutil.rmtree('IA')
    if os.path.exists('tests/Unit/AutoML/IA'):
        shutil.rmtree('tests/Unit/AutoML/IA')
