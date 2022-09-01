#  Copyright (c) 2022 by Amplo.

"""
Enables connection to Databricks via API calls.
"""


from __future__ import annotations

import json
import os
from copy import deepcopy

import requests

from amplo.utils import check_dtypes

__all__ = ["DatabricksJobsAPI", "train_on_cloud"]


class DatabricksJobsAPI:
    """
    Helper class for working with Databricks' Job API.

    References
    ----------
    [API reference](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/jobs)

    Parameters
    ----------
    host : str
        Databricks host (see https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/authentication).
    access_token : str
        Access token (see Databricks > User Settings > Access tokens).
    """

    def __init__(self, host: str, access_token: str):
        check_dtypes(("host", host, str), ("access_token", access_token, str))
        self.host = host
        self.access_token = access_token

    @classmethod
    def from_os_env(
        cls, host: str = None, access_token: str = None
    ) -> DatabricksJobsAPI:
        """
        Instantiate the class using os environment strings.

        Parameters
        ----------
        host : str, default: "DATABRICKS_HOST"
            Key in the os environment for the Databricks host.
        access_token : str, default: "DATABRICKS_TOKEN"
            Key in the os environment for the Databricks access token.

        Returns
        -------
        DatabricksJobsAPI
        """
        host = host or "DATABRICKS_HOST"
        access_token = access_token or "DATABRICKS_TOKEN"
        check_dtypes(("host", host, str), ("access_token", access_token, str))
        return cls(os.getenv(host), os.getenv(access_token))

    def __repr__(self):
        """
        Readable string representation of the class.
        """

        return f"{self.__class__.__name__}: <{self.host}>"

    def request(self, method: str, action: str, body: dict = None) -> dict:
        """
        Send a request to Databricks.

        Notes
        -----
        Every Databricks Jobs API call can be requested from here. In case you need more
        functionality than already implemented, have a look at the [OpenAPI
        specification](https://docs.microsoft.com/en-us/azure/databricks/_static/api-refs/jobs-2.1-azure.yaml).

        Parameters
        ----------
        method : str
            Request method (``GET``, ``PUT``, ``POST``, ...).
        action : str
            Path to Databricks Job (see Databricks Jobs API specification).
        body : dict of {str: int or str}, optional
            Request body schema (see Databricks Jobs API specification).

        Returns
        -------
        dict or {str: int or str}
            The request's response.

        Raises
        ------
        requests.HTTPError
            When the request's response has another status code than 200.
        """

        url = f"{self.host}/api/{action.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        body = body or {}

        response = requests.request(method, url, headers=headers, json=body)
        if not response.status_code == 200:
            raise requests.HTTPError(f"{response} {response.text}")

        return response.json()

    # --------------------------------------------------------------------------
    # Jobs API requests

    def list_jobs(
        self, limit: int = None, offset: int = None, expand_tasks: bool = None
    ) -> dict:
        body = {"limit": limit, "offset": offset, "expand_tasks": expand_tasks}
        return self.request("get", "2.1/jobs/list", body)

    def list_runs(
        self,
        limit: int = None,
        offset: int = None,
        expand_tasks: bool = None,
        *,
        active_only: bool = None,
        completed_only: bool = None,
        job_id: int = None,
        start_time_from: int = None,
        start_time_to: int = None,
    ) -> dict:
        body = {
            "limit": limit,
            "offset": offset,
            "expand_tasks": expand_tasks,
            "active_only": active_only,
            "completed_only": completed_only,
            "job_id": job_id,
            "start_time_from": start_time_from,
            "start_time_to": start_time_to,
        }
        return self.request("get", "2.1/jobs/runs/list", body)

    def get_job(self, job_id: int) -> dict:
        return self.request("get", "2.1/jobs/get", {"job_id": job_id})

    def get_run(self, run_id: int, include_history: bool = None) -> dict:
        body = {"run_id": run_id, include_history: include_history}
        return self.request("get", "2.1/jobs/runs/get", body)

    def run_job(
        self,
        job_id: int,
        idempotency_token: str = None,
        notebook_params: dict[str, int | str] = None,
        # Note: more parameters are available
    ) -> dict:
        body = {
            "job_id": job_id,
            "idempotency_token": idempotency_token,
            "notebook_params": notebook_params,
        }
        return self.request("post", "2.1/jobs/run-now", body)

    def cancel_run(self, run_id: int) -> dict:
        return self.request("post", "2.1/jobs/runs/cancel", {"run_id": run_id})


def train_on_cloud(
    job_id: int,
    model_version: int = 1,
    idempotency_token: str = None,
    notebook_params: dict[str, dict | bool | int | str] = None,
    *,
    host_key: str = None,
    access_token_key: str = None,
) -> dict[str, int]:
    """
    Starts a job run via the DatabricksJobsAPI.

    Notes
    -----
    Make sure to have set the following environment variables:
        - ``DATABRICKS_INSTANCE``
          (see https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/authentication).
        - ``DATABRICKS_ACCESS_TOKEN`` (see Databricks > User Settings > Access tokens).

    Note two important differences to ``DatabricksJobsAPI.run_job``.
    The "pipe_kwargs" key of ``notebook_params``:
        - will be JSON dumped to a string for you.
        - gets default values imputed if not given.

    Parameters
    ----------
    job_id : int
        Job ID in Databricks.
    model_version : int, default: 1
        Version of the model. Will overwrite the "version" key in the "pipe_kwargs" key
        of ``notebook_params``.
    idempotency_token : str, optional
        Idempotency token to ensure uniqueness of job run.
    notebook_params : dict of {str: bool or int or str}
        Parameters for job run. See widgets in notebook of corresponding job.
    *
    host_key : str, default: "DATABRICKS_HOST"
        Key in the os environment for the Databricks host.
    access_token_key : str, default: "DATABRICKS_TOKEN"
        Key in the os environment for the Databricks access token.

    Returns
    -------
    dict of {str: int}
        If response is success (200), ``run_id`` (globally unique key of newly triggered
        run) is one of the present keys.

    Raises
    ------
    ValueError
        When one or more widget / keywords in ``notebook_params`` is missing.
    """

    check_dtypes(
        ("job_id", job_id, int),
        ("model_version", model_version, int),
        ("idempotency_token", idempotency_token, (type(None), str)),
        ("notebook_params", notebook_params, dict),
    )
    notebook_params = deepcopy(notebook_params)

    # Define default keyword arguments and overwrite them when given by user input.
    pipe_kwargs = {
        "interval_analyse": False,
        "standardize": False,
        "missing_values": "zero",
        "balance": False,
        "stacking": False,
        "grid_search_time_budget": 7200,
        "n_grid_searches": 1,
        "verbose": 1,
    }
    pipe_kwargs.update(notebook_params.get("pipe_kwargs", {}))
    pipe_kwargs["version"] = model_version
    notebook_params["pipe_kwargs"] = json.dumps(pipe_kwargs)

    # Ensure presence of necessary keyword arguments / widgets
    widgets = ("team", "machine", "service", "issue", "model_id", "pipe_kwargs")
    missing = set(widgets) - set(notebook_params)
    if missing:
        msg = f"Missing keys for the parameter ``notebook_params``: {missing}"
        raise ValueError(msg)

    # Send request
    api = DatabricksJobsAPI.from_os_env(host_key, access_token_key)
    return api.run_job(job_id, idempotency_token, notebook_params)
