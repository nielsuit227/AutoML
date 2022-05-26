import pytest
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from Amplo import Pipeline
from Amplo.AutoML import Modeller


class TestPipeline:

    @pytest.mark.parametrize('mode', ['classification', 'regression'])
    @pytest.mark.parametrize('n_samples', [100, 100_000])
    def test_main_predictors(self, mode, make_x_y, n_samples):
        # Test mode
        x, y = make_x_y
        pipeline = Pipeline(grid_search_iterations=1, grid_search_candidates=1, plot_eda=False)
        pipeline.fit(x, y)
        for model in Modeller(mode=mode, samples=n_samples).return_models():
            print(model)
            x_c, _ = pipeline.convert_data(x)
            model.fit(x_c, y)
            pipeline.bestModel = model
            pipeline.predict(x)
            assert isinstance(pipeline._main_predictors, dict), 'Main predictors not dictionary.'

    def test_no_dirs(self):
        pipeline = Pipeline(no_dirs=True)
        assert not os.path.exists('AutoML'), 'Directory created'

    @pytest.mark.parametrize('mode', ['regression'])
    def test_no_args(self, mode, make_x_y):
        x, y = make_x_y
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline.fit(x, y)

    @pytest.mark.parametrize('mode', ['classification', 'regression'])
    def test_mode_detector(self, mode, make_x_y):
        x, y = make_x_y
        pipeline = Pipeline()
        pipeline._read_data(x, y)._mode_detector()
        assert pipeline.mode == mode

    @pytest.mark.parametrize('mode', ['classification'])
    def test_create_folders(self, mode, make_x_y):
        x, y = make_x_y
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline.fit(x, y)

        # Test Directories
        assert os.path.exists('AutoML')
        assert os.path.exists('AutoML/Data')
        assert os.path.exists('AutoML/Features')
        assert os.path.exists('AutoML/Production')
        assert os.path.exists('AutoML/Documentation')
        assert os.path.exists('AutoML/Results.csv')

    def test_read_write_csv(self):
        """
        Check whether intermediate data is stored and read correctly
        """
        # Set path
        data_path = 'test_data.csv'

        # Test single index
        data_write = pd.DataFrame(np.random.randint(0, 100, size=(10, 10)),
                                  columns=[f'feature_{i}' for i in range(10)])
        data_write.index.name = 'index'
        Pipeline._write_csv(data_write, data_path)
        data_read = Pipeline._read_csv(data_path)
        assert data_write.equals(data_read), 'Read data should be equal to original data'

        # Test multi-index (cf. IntervalAnalyser)
        data_write = data_write.set_index(data_write.columns[-2:].to_list())
        data_write.index.names = ['log', 'index']
        Pipeline._write_csv(data_write, data_path)
        data_read = Pipeline._read_csv(data_path)
        assert data_write.equals(data_read), 'Read data should be equal to original data'

        # Remove data
        os.remove(data_path)
