from . import GridSearchTestCase
from Amplo.GridSearch import OptunaGridSearch


class TestOptunaGridSearch(GridSearchTestCase):

    def test_small_regression(self):
        self._test_each_model(OptunaGridSearch, 'regression', 'r2', False)

    def test_big_regression(self):
        self._test_each_model(OptunaGridSearch, 'regression', 'r2', True)

    def test_small_classification(self):
        self._test_each_model(OptunaGridSearch, 'regression', 'r2', False)

    def test_big_classification(self):
        self._test_each_model(OptunaGridSearch, 'classification', 'accuracy', True)
