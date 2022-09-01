#  Copyright (c) 2022 by Amplo.

import json

import pytest
from requests import HTTPError

from amplo.api.databricks import DatabricksJobsAPI, train_on_cloud

DUMMY_JOB_ID = 749905062420360


class TestDatabricksJobsAPI:
    @classmethod
    def setup_class(cls):
        cls.api = DatabricksJobsAPI.from_os_env()

    def test_request(self):
        # A valid request should return a dictionary
        out = self.api.request("get", "2.1/jobs/list")
        assert isinstance(out, dict)
        # An invalid request should raise an error
        with pytest.raises(HTTPError):
            self.api.request("get", "")

    def test_list_jobs(self):
        expected_out = self.api.request("get", "2.1/jobs/list", {"limit": 1})
        actual_out = self.api.list_jobs(limit=1)
        assert expected_out == actual_out

    def test_list_runs(self):
        expected_out = self.api.request("get", "2.1/jobs/runs/list", {"limit": 1})
        actual_out = self.api.list_runs(limit=1)
        assert expected_out == actual_out

    def test_get_job(self):
        expected_out = self.api.request("get", "2.1/jobs/get", {"job_id": DUMMY_JOB_ID})
        actual_out = self.api.get_job(DUMMY_JOB_ID)
        assert expected_out == actual_out

    def test_get_run(self):
        # Find an existing run
        runs = self.api.list_runs(limit=1).get("runs")
        if len(runs) == 0:
            pytest.skip("There exists no run.")
        run_id = runs[0]["run_id"]

        # Test
        expected_out = self.api.request("get", "2.1/jobs/runs/get", {"run_id": run_id})
        actual_out = self.api.get_run(run_id)
        assert expected_out == actual_out

    def test_run_job(self):
        run_id = self.api.run_job(DUMMY_JOB_ID)["run_id"]
        assert self.api.cancel_run(run_id) == {}


def test_train_on_cloud():
    api = DatabricksJobsAPI.from_os_env()

    # Case 1: Missing keys
    with pytest.raises(ValueError):
        train_on_cloud(DUMMY_JOB_ID, notebook_params={})

    # Case 2: Key "pipe_kwargs" is not provided but shall be set by `train_on_cloud`
    params = {"team": 0, "machine": 0, "service": 0, "issue": 0, "model_id": 0}
    job = train_on_cloud(DUMMY_JOB_ID, notebook_params=params)
    run = api.get_run(job["run_id"])
    api.cancel_run(run["run_id"])

    passed_params = run["overriding_parameters"]["notebook_params"]
    assert set(params) | {"pipe_kwargs"} == set(passed_params)

    # Case 3: Key "pipe_kwargs" is present but default parameters shall be set
    params["pipe_kwargs"] = {"verbose": 2}
    job = train_on_cloud(DUMMY_JOB_ID, notebook_params=params)
    run = api.get_run(job["run_id"])
    api.cancel_run(run["run_id"])

    passed_params = run["overriding_parameters"]["notebook_params"]
    passed_params["pipe_kwargs"] = json.loads(passed_params["pipe_kwargs"])
    assert params["pipe_kwargs"]["verbose"] == passed_params["pipe_kwargs"]["verbose"]
    assert params.pop("pipe_kwargs") != passed_params.pop("pipe_kwargs")
    params = {key: str(params[key]) for key in params}  # requests stringifies integers
    assert params == passed_params
