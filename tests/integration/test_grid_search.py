import pytest

import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, StratifiedKFold

from Amplo import Pipeline
from Amplo.AutoML import Modeller
from Amplo.GridSearch import BaseGridSearch, HalvingGridSearch, OptunaGridSearch
from Amplo.Utils.io import parse_json


@pytest.fixture(scope='class', params=['regression', 'classification'])
def make_data(request):
    mode = request.param

    if mode == 'regression':
        request.cls.k_fold = KFold
        request.cls.objective = 'r2'
        x, y = make_regression()
    elif mode == 'classification':
        request.cls.k_fold = StratifiedKFold
        request.cls.objective = 'r2'
        x, y = make_classification()
    else:
        raise ValueError('Mode is invalid')

    request.cls.mode = mode
    request.cls.x = pd.DataFrame(x)
    request.cls.y = pd.Series(y)
    yield


@pytest.mark.usefixtures('make_data')
class TestGridSearch:

    @pytest.mark.parametrize('test_case', ['grid_search_type', 'grid_search_iterations'])
    def test_no_grid_search(self, test_case):
        # Set keyword arguments
        kwargs = dict(extract_features=False, sequence=False, plot_eda=False)
        if test_case == 'grid_search_type':
            kwargs.update({test_case: None})
        elif test_case == 'grid_search_iterations':
            kwargs.update({test_case: 0})
        else:
            raise ValueError('Invalid test case')
        # Fit pipeline
        pipeline = Pipeline(**kwargs)
        pipeline.fit(self.x, self.y)
        # Check results
        assert any(pipeline.results.loc[:, 'type'] == 'Hyper Parameter'), 'No hyper parameter results were found'

    @pytest.mark.parametrize('grid_search', [BaseGridSearch, HalvingGridSearch, OptunaGridSearch])
    @pytest.mark.parametrize('simulate', ['small', 'big'])
    def test_each_model(self, grid_search, simulate):
        """
        Given the models which depend on 1) mode and 2) num_samples,
        each model is trained on the given grid search class.

        Thanks to the fact that all grid search models have the same interface,
        this is possible :-)
        """

        if simulate == 'small':
            n_samples = 100
        elif simulate == 'big':
            n_samples = 50_000
        else:
            raise ValueError('Invalid parameter')

        modeller_kwargs = dict(
            mode=self.mode,
            objective=self.objective,
            samples=n_samples,
        )
        models = Modeller(**modeller_kwargs).return_models()

        for model in models:
            # Grid search
            search = grid_search(model,
                                 cv=self.k_fold(n_splits=3),
                                 verbose=0,
                                 timeout=10,
                                 candidates=2,
                                 scoring=self.objective)
            results = search.fit(self.x, self.y)

            # Tests
            model_name = type(model).__name__
            assert isinstance(results, pd.DataFrame), f'Expected result to be pandas.DataFrame' \
                                                      f'but found {type(results)} in {model_name} instead'
            results_keys = ['worst_case', 'mean_objective', 'std_objective', 'params', 'mean_time', 'std_time']
            assert all(key in results.keys() for key in results_keys), f'Keys are missing in {model_name}'
            assert len(results) > 0, f'Results are empty in {model_name}'
            model.set_params(**parse_json(results.iloc[0]['params']))
