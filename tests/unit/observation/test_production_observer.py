#  Copyright (c) 2022 by Amplo.

import pytest

from amplo import Pipeline
from amplo.observation import ProductionObserver
from amplo.observation._base import ProductionWarning
from tests import DelayedRandomPredictor


class TestProductionObserver:
    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_check_prediction_latency(self, mode, make_x_y):
        pytest.skip()

        x, y = make_x_y

        # Make pipeline and simulate fit
        pipeline = Pipeline(n_grid_searches=0)
        pipeline._read_data(x, y)
        pipeline._mode_detector()
        pipeline.best_model = DelayedRandomPredictor(delay=0.1, mode=mode)
        pipeline.best_model.fit(x, y)

        # Observe
        obs = ProductionObserver(pipeline)
        with pytest.warns(ProductionWarning):
            obs.check_prediction_latency(threshold=0.1)
