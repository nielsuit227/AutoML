import unittest

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from Amplo.AutoML import Modeller
from Amplo.GridSearch._GridSearch import _GridSearch  # noqa
from Amplo.Utils.io import parse_json


__all__ = ['GridSearchTestCase']


class GridSearchTestCase(unittest.TestCase):
    """Base class for Grid Search testing"""

    @classmethod
    def setUpClass(cls) -> None:
        x, y = make_classification()
        cls.cx = pd.DataFrame(x)
        cls.cy = pd.Series(y)
        x, y = make_regression()
        cls.rx = pd.DataFrame(x)
        cls.ry = pd.Series(y)

    def _test_each_model(
            self,
            grid_search: _GridSearch.__class__,
            mode: str,
            objective: str,
            simulate_big: bool,
    ):
        """
        Given the models which depend on 1) mode and 2) num_samples,
        each model is trained on the given grid search class.

        Thanks to the fact that all grid search models have the same interface,
        this is possible :-)

        Parameters
        ----------
        grid_search : _GridSearch.__class__
            A grid search class that inherited from `Amplo.GridSearch._GridSearch`
        mode : str
            Training mode. Either 'regression' or 'classification'
        objective : str
            An objective that is valid for `Amplo.AutoML.Modeller`
        simulate_big : bool
            Whether to fabricate models that were intended for large datasets.
        """

        if mode == 'regression':
            k_fold = KFold
            data_x, data_y = self.rx, self.ry
        elif mode == 'classification':
            k_fold = StratifiedKFold
            data_x, data_y = self.cx, self.cy
        else:
            raise ValueError('Mode is invalid')

        num_samples = 50_000 if simulate_big else 100

        models = Modeller(mode=mode, objective=objective, samples=num_samples).return_models()
        for model in models:
            # Grid search
            search = grid_search(model,
                                 cv=k_fold(n_splits=3),
                                 verbose=0,
                                 timeout=10,
                                 candidates=2,
                                 scoring=objective)
            results = search.fit(data_x, data_y)

            # Tests
            model_name = type(model).__name__
            assert isinstance(results, pd.DataFrame), f'Expected result to be pandas.DataFrame' \
                                                      f'but found {type(results)} in {model_name} instead'
            results_keys = ['worst_case', 'mean_objective', 'std_objective', 'params', 'mean_time', 'std_time']
            assert all(key in results.keys() for key in results_keys), f'Keys are missing in {model_name}'
            assert len(results) > 0, f'Results are empty in {model_name}'
            model.set_params(**parse_json(results.iloc[0]['params']))
