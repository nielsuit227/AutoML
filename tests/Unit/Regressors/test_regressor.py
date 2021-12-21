from sklearn.base import clone
from sklearn.metrics import SCORERS


class TestRegression(object):

    def __init__(self):
        self.model = None
        self.x = None
        self.y = None

    def test_set_params(self):
        self.model.set_params(**{'max_depth': 10})

    def test_get_params(self):
        self.model.get_params()

    def test_pd_fit(self):
        model = clone(self.model)
        model.fit(self.x, self.y)

    def test_np_fit(self):
        model = clone(self.model)
        model.fit(self.x.to_numpy(), self.y.to_numpy())

    def test_trained_attr(self):
        model = clone(self.model)
        assert hasattr(model, 'trained')
        assert model.trained is False
        model.fit(self.x, self.y)
        assert model.trained is True

    def test_predict(self):
        model = clone(self.model)
        model.fit(self.x, self.y)
        prediction = model.predict(self.x)

        assert len(prediction.shape) == 1

    def test_cloneable(self):
        params = {'max_depth': 5}
        cloned = clone(clone(self.model).set_params(**params))
        cloned.fit(self.x, self.y)
        prediction = cloned.predict(self.x)

        assert len(prediction.shape) == 1

    def test_scorer(self):
        model = clone(self.model)
        model.fit(self.x, self.y)
        SCORERS['neg_mean_absolute_error'](model, self.x, self.y)
        SCORERS['neg_mean_squared_error'](model, self.x, self.y)
        SCORERS['r2'](model, self.x, self.y)

