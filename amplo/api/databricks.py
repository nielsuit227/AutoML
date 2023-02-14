#  Copyright (c) 2022 by Amplo.

"""
Enables connection to Databricks via API calls.
"""

from __future__ import annotations

import os
from typing import Any

from typing_extensions import Self

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

    def _authorization_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    @classmethod
    def from_os_env(
        cls, host_os: str | None = None, access_token_os: str | None = None
    ) -> Self:
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

        Raises
        ------
        KeyError
            When a os variable is not set.
        """

        host_os = host_os or _DATABRICKS_HOST_OS
        access_token_os = access_token_os or _DATABRICKS_TOKEN_OS

        host = os.environ[host_os]
        access_token = os.environ[access_token_os]
        return cls(host, access_token)

    def request_json(
        self, method: str, action: str, body: dict[str, Any] | None = None
    ) -> dict[str, int | str]:
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
        self,
        limit: int | None = None,
        offset: int | None = None,
        expand_tasks: bool | None = None,
    ) -> dict[str, int | str]:
        body = {"limit": limit, "offset": offset, "expand_tasks": expand_tasks}
        return self.request_json("get", "2.1/jobs/list", body)

    def list_runs(
        self,
        limit: int | None = None,
        offset: int | None = None,
        expand_tasks: bool | None = None,
        *,
        active_only: bool | None = None,
        completed_only: bool | None = None,
        job_id: int | None = None,
        start_time_from: int | None = None,
        start_time_to: int | None = None,
    ) -> dict[str, Any]:
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
        return self.request_json("get", "2.1/jobs/runs/list", body)

    def get_job(self, job_id: int) -> dict[str, int | str]:
        return self.request_json("get", "2.1/jobs/get", {"job_id": job_id})

    def get_run(
        self, run_id: int, include_history: bool | None = None
    ) -> dict[str, str | int]:
        body = {"run_id": run_id, "include_history": include_history}
        return self.request_json("get", "2.1/jobs/runs/get", body)

    def run_job(
        self,
        job_id: int,
        idempotency_token: str | None = None,
        notebook_params: dict[str, Any] | None = None,
        # Note: more parameters are available
    ) -> dict[str, Any]:
        body = {
            "job_id": job_id,
            "idempotency_token": idempotency_token,
            "notebook_params": notebook_params,
        }
        return self.request_json("post", "2.1/jobs/run-now", body)

    def cancel_run(self, run_id: int) -> dict[str, int | str]:
        return self.request_json("post", "2.1/jobs/runs/cancel", {"run_id": run_id})
