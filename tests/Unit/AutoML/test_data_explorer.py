import os
import pytest
import shutil
import unittest
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from Amplo.AutoML import DataExplorer


class TestDataExploring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.class_x, cls.class_y = load_iris(return_X_y=True, as_frame=True)
        cls.reg_x, cls.reg_y = fetch_california_housing(return_X_y=True)

    def test_regression(self):
        eda = DataExplorer(self.reg_x, y=self.reg_y, mode='regression', folder='tmp/')
        eda.run()

    def test_classification(self):
        eda = DataExplorer(self.class_x, y=self.class_y, mode='classification', folder='tmp/')
        eda.run()


@pytest.fixture(scope="session", autouse=True)
def teardown():
    yield
    if os.path.exists('tmp'):
        shutil.rmtree('tmp')
