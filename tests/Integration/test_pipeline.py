import os
import json
import joblib
import shutil
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss

from Amplo import Pipeline
from Amplo.Utils.testing import make_interval_data
from Amplo.AutoML import IntervalAnalyser
from tests import rmtree_automl


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x, y = make_classification()
        cls.c_data = pd.DataFrame(x, columns=['Feature_{}'.format(i) for i in range(x.shape[1])])
        cls.c_data['target'] = y
        x, y = make_regression()
        cls.r_data = pd.DataFrame(x, columns=['Feature_{}'.format(i) for i in range(x.shape[1])])
        cls.r_data['target'] = y

    @rmtree_automl
    def test_mode_detector(self):
        # Classification numeric
        pipeline = Pipeline(no_dirs=True, target='target')
        pipeline._mode_detector(self.c_data)
        assert pipeline.mode == 'classification'

        # Classification categorical
        df = self.c_data
        df['target'] = [f'Class_{v}' for v in df['target'].values]
        pipeline = Pipeline(no_dirs=True, target='target')
        pipeline._mode_detector(self.c_data)
        assert pipeline.mode == 'classification'

        # Regression
        pipeline = Pipeline(no_dirs=True, target='target')
        pipeline._mode_detector(self.r_data)
        assert pipeline.mode == 'regression'

    @rmtree_automl
    def test_stacking(self):
        pipeline = Pipeline(target='target', grid_search_candidates=1,
                            stacking=True, feature_timeout=5)
        pipeline.fit(self.c_data)
        pipeline._prepare_production_files(model='StackingClassifier')
        shutil.rmtree('AutoML')
        pipeline = Pipeline(target='target', grid_search_candidates=1,
                            stacking=True, feature_timeout=5)
        pipeline.fit(self.r_data)
        pipeline._prepare_production_files(model='StackingRegressor')
        shutil.rmtree('AutoML')

    @rmtree_automl
    def test_regression(self):
        pipeline = Pipeline(target='target',
                            name='AutoReg',
                            mode='regression',
                            objective='r2',
                            feature_timeout=5,
                            grid_search_iterations=0,
                            plot_eda=False,
                            process_data=False,
                            document_results=False,
                            )
        pipeline.fit(self.r_data)

        # Test Directories
        assert os.path.exists('AutoML')
        assert os.path.exists('AutoML/Data')
        assert os.path.exists('AutoML/Features')
        assert os.path.exists('AutoML/Production')
        assert os.path.exists('AutoML/Documentation')
        assert os.path.exists('AutoML/Results.csv')

        # Test data handling
        c, _ = pipeline.convert_data(self.r_data.drop('target', axis=1))
        x = pipeline.x[pipeline.settings['features']]
        y = self.r_data['target']
        assert np.allclose(c.values, x.values), "Inconsistent X: max diff: {:.2e}"\
            .format(np.max(abs(c.values - x.values)))
        assert np.allclose(y, pipeline.y), "Inconsistent Y"

        # Pipeline Prediction
        prediction = pipeline.predict(self.r_data)
        assert len(prediction.shape) == 1
        assert r2_score(self.r_data['target'], prediction) > 0.75

        # Settings prediction
        settings = json.load(open('AutoML/Production/v1/Settings.json', 'r'))
        model = joblib.load('AutoML/Production/v1/Model.joblib')
        p = Pipeline(no_dirs=True)
        p.load_settings(settings)
        p.load_model(model)
        assert np.allclose(p.predict(self.r_data), prediction)

        # Check settings
        assert 'pipeline' in settings
        assert 'version' in settings
        assert 'name' in settings['pipeline']
        assert 'model' in settings
        assert 'amplo_version' in settings
        assert 'params' in settings
        assert 'drift_detector' in settings
        assert 'features' in settings
        assert 'validation' in settings
        assert 'data_processing' in settings

    @rmtree_automl
    def test_classification(self):
        pipeline = Pipeline(target='target',
                            name='AutoClass',
                            mode='classification',
                            objective='neg_log_loss',
                            standardize=True,
                            feature_timeout=5,
                            grid_search_iterations=0,
                            plot_eda=False,
                            process_data=True,
                            document_results=False,
                            )
        pipeline.fit(self.c_data)

        # Tests
        assert os.path.exists('AutoML')
        assert os.path.exists('AutoML/EDA')
        assert os.path.exists('AutoML/Data')
        assert os.path.exists('AutoML/Features')
        assert os.path.exists('AutoML/Production')
        assert os.path.exists('AutoML/Results.csv')

        # Pipeline Prediction
        prediction = pipeline.predict_proba(self.c_data)
        assert log_loss(self.c_data['target'], prediction) > -1

        # Settings prediction
        settings = json.load(open('AutoML/Production/v1/Settings.json', 'r'))
        model = joblib.load('AutoML/Production/v1/Model.joblib')
        p = Pipeline(no_dirs=True)
        p.load_settings(settings)
        p.load_model(model)
        assert np.allclose(p.predict_proba(self.c_data), prediction)

        # Check settings
        assert 'pipeline' in settings
        assert 'version' in settings
        assert 'model' in settings
        assert 'amplo_version' in settings
        assert 'params' in settings
        assert 'drift_detector' in settings
        assert 'features' in settings
        assert 'validation' in settings
        assert 'data_processing' in settings

    def test_drift(self):
        # todo connect with logger
        warnings.warn('This test is not yet implemented')
        pass

    def test_all_models(self):
        # TODO
        warnings.warn('This test is not yet implemented')
        pass

    @rmtree_automl
    def test_interval_analyzer(self):
        """Use interval analyzer by default when a folder with logs is presented to pipeline"""
        # Create dummy data
        data_dir = './test_data'
        make_interval_data(directory=data_dir)
        # TODO: does it have to work also with n_labels=1 and n_logs=1?

        # Pipeline
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline.fit(data_dir)

        # Check initial data handling
        assert pipeline._data_from_interval_analyzer, '``IntervalAnalyser`` was not activated'
        assert Path('AutoML/Data/Interval_Analyzed_v1.csv').exists(), \
            'Interval-analyzed data was not properly stored'
        assert pipeline.target == IntervalAnalyser.target, 'Targets should match'
        assert list(pipeline.data.index.names) == ['log', 'index'], 'Index is incorrect'

        # Check data processing
        assert Path('AutoML/Data/Cleaned_Interval_v1.csv').exists(), \
            'Either ``IntervalAnalyser`` did not call ``Pipeline._data_processing()`` ' \
            'or *.csv file was not stored / stored in different path'
        assert pipeline.dataProcessor.is_fitted, 'Data Processor was not fitted'

        # Remove dummy data
        shutil.rmtree(data_dir)
