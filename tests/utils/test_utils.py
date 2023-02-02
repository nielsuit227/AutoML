#  Copyright (c) 2022 by Amplo.

import numpy as np
import pytest

from amplo.automl.modelling import Modeller, get_model
from amplo.utils import check_dtypes, clean_feature_name, hist_search


class TestUtils:
    def test_get_model(self):
        # Test valid models
        for mode in ("classification", "regression"):
            for samples in [100, 100_000]:
                all_models = set(
                    type(model)
                    for model in Modeller(mode=mode, samples=samples).return_models()
                )
                for model_type in all_models:
                    model_name = model_type.__name__
                    model = get_model(model_name)
                    assert type(model) == model_type

        # Test invalid model
        with pytest.raises(ValueError):
            get_model("ImaginaryModel")

    def test_hist_search(self):
        bin_idx = hist_search(np.arange(100).astype(float), 50)
        assert bin_idx == 50, "Returned wrong bin index."

    def test_clean_feature_name(self):
        ugly = "   This-is (an)UGLY   [ string__"
        pretty = "this_is_an_ugly_string"
        assert pretty == clean_feature_name(ugly)

    def test_check_dtypes(self):
        # Test valid dtypes
        check_dtypes(("1", 1, int), ("2.0", 2.0, float), ("both", 3.0, (int, float)))
        # Test invalid dtypes
        with pytest.raises(TypeError):
            check_dtypes(("invalid", 1.0, int))
