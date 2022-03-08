import os
import shutil
import unittest
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from Amplo import Pipeline
from Amplo.AutoML import Modeller


class TestPipeline(unittest.TestCase):

    def test_main_predictors(self):
        if os.path.exists('AutoML'):
            shutil.rmtree('AutoML')
        for mode in ['classification', 'regression']:
            if mode == 'classification':
                x, y = make_classification()
            else:
                x, y = make_regression()
            x, y = pd.DataFrame(x), pd.Series(y)
            pipeline = Pipeline(grid_search_iterations=1, grid_search_candidates=1, plot_eda=False)
            pipeline.fit(x, y)
            for samples in [100, 25001]:
                for model in Modeller(mode=mode, samples=samples).return_models():
                    print(model)
                    x_c, _ = pipeline.convert_data(x)
                    model.fit(x_c, y)
                    pipeline.bestModel = model
                    pipeline.predict(x)
                    assert isinstance(pipeline._main_predictors, dict), 'Main predictors not dictionary.'

            if os.path.exists('AutoML'):
                shutil.rmtree('AutoML')

    def test_no_dirs(self):
        if os.path.exists('AutoML'):
            shutil.rmtree('AutoML')
        pipeline = Pipeline(no_dirs=True)
        assert not os.path.exists('AutoML'), 'Directory created'

    def test_no_args(self):
        if os.path.exists('AutoML'):
            shutil.rmtree('AutoML')
        x, y = make_regression()
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline.fit(x, y)
        shutil.rmtree('AutoML')

    def test_mode_detector(self):
        x, y = make_regression()
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline.fit(x, y)
        assert pipeline.mode == 'regression'
        shutil.rmtree('AutoML')
        x, y = make_classification()
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline.fit(x, y)
        assert pipeline.mode == 'classification'
        shutil.rmtree('AutoML')

    def test_create_folders(self):
        if os.path.exists('AutoML'):
            shutil.rmtree('AutoML')

        x, y = make_classification()
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline.fit(x, y)

        # Test Directories
        assert os.path.exists('AutoML')
        assert os.path.exists('AutoML/Data')
        assert os.path.exists('AutoML/Features')
        assert os.path.exists('AutoML/Production')
        assert os.path.exists('AutoML/Documentation')
        assert os.path.exists('AutoML/Results.csv')
