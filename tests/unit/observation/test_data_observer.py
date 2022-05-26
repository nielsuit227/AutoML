import json
import pytest
import re

import numpy as np

from Amplo import Pipeline
from Amplo.Observation._data_observer import DataObserver
from Amplo.Observation.base import ProductionWarning


class TestModelObserver:

    def test_monotonic_columns(self):
        size = 100
        monotonic = np.array(range(-10, size - 10)) * 4.2  # start=-10, step=4.2
        random = np.random.normal(size=size)
        x = np.concatenate([monotonic[:, None], random[:, None]], axis=1)
        y = random  # does not matter

        # Observe
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline._read_data(x, y)
        obs = DataObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning) as record:
            obs.check_monotonic_columns()
        msg = str(record[0].message)
        monotonic_cols = json.loads(re.search(r"\[.*]", msg).group(0))
        assert monotonic_cols == [0], "Wrong monotonic columns identified."
