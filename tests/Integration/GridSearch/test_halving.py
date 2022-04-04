import warnings

from . import GridSearchTestCase
from Amplo.GridSearch import HalvingGridSearch


warnings.filterwarnings('ignore')


class TestHalvingGridSearch(GridSearchTestCase):

    def test_small_regression(self):
        self._test_each_model(HalvingGridSearch, 'regression', 'r2', False)

    def test_big_regression(self):
        self._test_each_model(HalvingGridSearch, 'regression', 'r2', True)

    def test_small_classification(self):
        self._test_each_model(HalvingGridSearch, 'regression', 'r2', False)

    def test_big_classification(self):
        self._test_each_model(HalvingGridSearch, 'classification', 'accuracy', True)
