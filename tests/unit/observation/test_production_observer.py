#  Copyright (c) 2022 by Amplo.

import pytest

from amplo import Pipeline
from amplo.observation._base import ProductionWarning
from tests import DelayedRandomPredictor


class TestProductionObserver:
    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_check_prediction_latency(self, mode, make_x_y):
        pytest.skip()
