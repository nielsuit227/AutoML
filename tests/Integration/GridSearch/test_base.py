from . import GridSearchTestCase
from Amplo.GridSearch import BaseGridSearch


class TestBaseGridSearch(GridSearchTestCase):

    def test_small_regression(self):
        self._test_each_model(BaseGridSearch, 'regression', 'r2', False)

    def test_big_regression(self):
        self._test_each_model(BaseGridSearch, 'regression', 'r2', True)

    def test_small_classification(self):
        self._test_each_model(BaseGridSearch, 'regression', 'r2', False)

    def test_big_classification(self):
        self._test_each_model(BaseGridSearch, 'classification', 'accuracy', True)
