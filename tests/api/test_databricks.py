#  Copyright (c) 2022 by Amplo.

from inspect import signature

import pytest
from requests import HTTPError

from amplo.api.databricks import DatabricksJobsAPI

DUMMY_JOB_ID = 749905062420360


class TestDatabricksJobsAPI:
    """
    Tests for the class `DatabricksJobsAPI`.
    """

    api: DatabricksJobsAPI

    @classmethod
    def setup_class(cls):
        cls.api = DatabricksJobsAPI.from_os_env()

    def test_request(self):
        # A valid request should return a dictionary
        out = self.api.request_json("get", "2.1/jobs/list")
        assert isinstance(out, dict)
        # An invalid request should raise an error
        with pytest.raises(HTTPError):
            self.api.request("get", "")

    def test_list_jobs(self):
        expected_out = self.api.request_json("get", "2.1/jobs/list", {"limit": 1})
        actual_out = self.api.list_jobs(limit=1)
        assert expected_out == actual_out

    def test_list_runs(self):
        expected_out = self.api.request_json("get", "2.1/jobs/runs/list", {"limit": 1})
        actual_out = self.api.list_runs(limit=1)
        assert expected_out == actual_out

    def test_get_job(self):
        expected_out = self.api.request_json(
            "get", "2.1/jobs/get", {"job_id": DUMMY_JOB_ID}
        )
        actual_out = self.api.get_job(DUMMY_JOB_ID)
        assert expected_out == actual_out

    def test_get_run(self):
        # Find an existing run
        runs = self.api.list_runs(limit=1).get("runs")
        if not runs or len(runs) == 0:
            pytest.skip("There exists no run.")
        run_id = runs[0]["run_id"]

        # Test
        expected_out = self.api.request_json(
            "get", "2.1/jobs/runs/get", {"run_id": run_id}
        )
        actual_out = self.api.get_run(run_id)
        assert expected_out == actual_out

    def test_run_job(self):
        run_id = self.api.run_job(DUMMY_JOB_ID)["run_id"]
        assert self.api.cancel_run(run_id) == {}


class TestDatabricksTrainOnCloudNotebook:
    """
    Test functionalities that are used for `train_on_cloud.py` notebook.

    References
    ----------
    - https://github.com/Amplo-AG/Databricks/blob/main/workflows/train_on_cloud.py
    """

    def test_pipeline_format(self):
        # Assert that `Pipeline` is importable
        from amplo import Pipeline

        # Assert that `__init__` accepts those keyword arguments and more
        init_params = signature(Pipeline).parameters.keys()
        expected_init_params = ["main_dir", "name"]
        assert set(init_params).issuperset(expected_init_params)
        assert len(init_params) > len(expected_init_params)  # want to pass more kwargs

    def test_upload_models_format(self):
        # Assert that `upload_model` is importable
        from amplo.api.platform import upload_model

        # Assert that `upload_models` is present and accepts those keyword arguments
        func_params = signature(upload_model).parameters.keys()
        expected_func_params = [
            "train_id",
            "model_dir",
            "team",
            "machine",
            "service",
            "issue",
            "version",
            "host",
            "access_token_os",
        ]
        assert set(expected_func_params) == set(func_params)
