#  Copyright (c) 2022 by Amplo.

import pytest
from sklearn.datasets import load_iris, make_regression

from amplo.automl import DataExplorer
from tests import rmtree


@pytest.fixture(scope="class", params=["classification", "regression"])
def make_mode(request):
    mode = request.param
    if mode == "classification":
        x, y = load_iris(return_X_y=True, as_frame=True)
    elif mode == "regression":
        # Interesting enough, the regression test for `test_explorer` ends up in a
        # deadlock, specifically in the `DataExplorer.shap()` method. It's even more
        # interesting that the california housing dataset works fine when using
        # `unittest`
        # x, y = fetch_california_housing(return_X_y=True, as_frame=True)
        x, y = make_regression()
    else:
        raise NotImplementedError("Invalid mode")
    request.cls.x = x
    request.cls.y = y
    request.cls.mode = mode
    yield


@pytest.mark.usefixtures("make_mode")
class TestDataExploring:
    def test_explorer(self):
        tmp_folder = "tmp/"
        eda = DataExplorer(self.x, y=self.y, mode=self.mode, folder=tmp_folder)
        eda.run()
        rmtree(tmp_folder)


# class TestDataExploring(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.class_x, cls.class_y = load_iris(return_X_y=True, as_frame=True)
#         cls.reg_x, cls.reg_y = fetch_california_housing(return_X_y=True)
#
#     def test_regression(self):
#         eda = DataExplorer(self.reg_x, y=self.reg_y, mode='regression', folder='tmp/')
#         eda.run()
#         rmtree('tmp/')
#
#     def test_classification(self):
#         eda = DataExplorer(self.class_x, y=self.class_y, mode='classification', folder='tmp/')  # noqa: E501
#         eda.run()
#         rmtree('tmp/')
