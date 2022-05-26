import pytest

import numpy as np

from Amplo import Pipeline
from Amplo.Observation._model_observer import ModelObserver
from Amplo.Observation.base import ProductionWarning
from tests import RandomPredictor


@pytest.fixture
def make_one_to_one_data(mode):
    size = 100
    if mode == 'classification':
        linear_col = np.random.choice([0, 1, 2], size)
    elif mode == 'regression':
        linear_col = np.random.uniform(0.0, 1.0, size)
    else:
        raise ValueError('Invalid mode')
    x = linear_col.reshape(-1, 1)
    y = linear_col.reshape(-1)
    yield x, y


class TestModelObserver:

    @pytest.mark.parametrize('mode', ['classification', 'regression'])
    def test_better_than_linear(self, mode, make_one_to_one_data):
        x, y = make_one_to_one_data

        # Make pipeline and simulate fit
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline._read_data(x, y)
        pipeline._mode_detector()
        pipeline.bestModel = RandomPredictor(mode=mode)

        # Observe
        obs = ModelObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning):
            obs.check_better_than_linear()
