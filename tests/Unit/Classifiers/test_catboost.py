import unittest
from sklearn.datasets import load_iris
from Amplo.Classifiers import CatBoostClassifier


class TestCatBoostClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = load_iris(return_X_y=True, as_frame=True)

    def test_set_params(self):
        model = CatBoostClassifier()
        model.set_params({'depth': 200})

    def test_get_params(self):
        model = CatBoostClassifier()
        model.get_params()

    def test_pd_fit(self):
        model = CatBoostClassifier()
        model.fit(self.x, self.y)

    def test_np_fit(self):
        model = CatBoostClassifier()
        model.fit(self.x.to_numpy(), self.y.to_numpy())

    def test_trained_attr(self):
        model = CatBoostClassifier
        assert hasattr(model, 'trained')
        assert model.trained is False
        model.fit(self.x, self.y)
        assert model.trained is True