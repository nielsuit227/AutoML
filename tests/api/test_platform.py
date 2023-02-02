#  Copyright (c) 2022 by Amplo.

import pytest
import requests

from amplo.api.platform import AmploPlatformAPI, _format_version


class TestAmploPlatformAPI:
    """
    Tests for the class `AmploPlatformAPI`.
    """

    api: AmploPlatformAPI

    @classmethod
    def setup_class(cls):
        cls.api = AmploPlatformAPI.from_os_env()

    def test_request(self):
        # A valid request should return a Response object
        out = self.api.request("get", "models", params={"team": "Demo"})
        assert isinstance(out, requests.Response)
        # An invalid request should raise an error
        with pytest.raises(requests.HTTPError):
            self.api.request("get", "does_not_exist")

    def test_list_models(self):
        params = {"team": "Demo"}
        expected_out = self.api.request("get", "models", params=params).json()
        actual_out = self.api.list_models(**params)
        assert expected_out == actual_out

    def test_list_trainings(self):
        params = {"team": "Demo", "service": "Diagnostics", "category": "Diagnostics"}
        expected_out = self.api.request("get", "trainings", params=params).json()
        actual_out = self.api.list_trainings(**params)
        assert expected_out == actual_out

    def test_upload_trainings(self):
        pytest.skip("Test not yet implemented. We need a dummy training_id for that.")


def test_format_version():
    """
    Test the function `_format_version`.
    """

    # Case 1: integer
    assert _format_version(1) == "v1"

    # Case 2: string
    assert _format_version("1") == "v1"
    assert _format_version("v1") == "v1"

    # Case 3: bad string format
    with pytest.raises(ValueError):
        _format_version("abc")
