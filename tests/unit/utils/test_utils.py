#  Copyright (c) 2022 by Amplo.

import numpy as np
import pytest

from amplo.automl.modelling import Modeller
from amplo.utils import check_dtypes, clean_feature_name, get_model, hist_search


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
                    model = get_model(model_name, mode=mode, samples=samples)
                    assert type(model) == model_type

        # Test invalid model
        try:
            get_model("ImaginaryModel", mode="regression", samples=100)
            raise AssertionError("`get_model` did not raise a ValueError.")
        except ValueError:
            pass

    def test_hist_search(self):
        bin_idx = hist_search(np.arange(100).astype(float), 50)
        assert bin_idx == 50, "Returned wrong bin index."

    def test_clean_feature_name(self):
        ugly = "   This-is (an)UGLY   [ string__"
        pretty = "this_is_an_ugly_string"
        assert pretty == clean_feature_name(ugly)

    def test_check_dtypes(self):
        # Test valid dtypes
        valid = [("1", 1, int), ("2.0", 2.0, float), ("both", 3.0, (int, float))]
        check_dtypes(*valid)
        # Test invalid dtypes
        invalid = [("invalid", 1.0, int)]
        with pytest.raises(TypeError):
            check_dtypes(*invalid)
