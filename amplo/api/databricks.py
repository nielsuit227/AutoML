#  Copyright (c) 2022 by Amplo.

"""
Enables connection to Databricks via API calls.
"""


from __future__ import annotations

import os

from amplo.api._base import BaseRequestAPI

__all__ = ["DatabricksJobsAPI"]


_DATABRICKS_HOST_OS = "DATABRICKS_HOST"
_DATABRICKS_TOKEN_OS = "DATABRICKS_TOKEN"


class DatabricksJobsAPI(BaseRequestAPI):
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
        super().__init__(host, access_token)

    def _authorization_header(self) -> dict:
        return {"Authorization": f"Bearer {self.access_token}"}

    @classmethod
    def from_os_env(
        cls, host_os: str = None, access_token_os: str = None
    ) -> DatabricksJobsAPI:
        """
        Instantiate the class using os environment strings.

        Parameters
        ----------
        host_os : str, default: _DATABRICKS_HOST_OS
            Key in the os environment for the Databricks host.
        access_token_os : str, default: _DATABRICKS_TOKEN_OS
            Key in the os environment for the Databricks access token.

        Returns
        -------
        DatabricksJobsAPI
        """

        host_os = host_os or _DATABRICKS_HOST_OS
        access_token_os = access_token_os or _DATABRICKS_TOKEN_OS
        return cls(os.getenv(host_os), os.getenv(access_token_os))

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

        return super().request(method, action, json=body).json()

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
