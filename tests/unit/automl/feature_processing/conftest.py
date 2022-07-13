#  Copyright (c) 2022 by Amplo.

import numpy as np
import pytest


@pytest.fixture(scope="class")
def make_rng(request):
    request.cls.rng = np.random.default_rng(seed=92938)
    yield
