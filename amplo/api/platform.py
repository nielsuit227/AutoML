#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, BinaryIO
from warnings import warn

from requests import Response
from typing_extensions import Self

from amplo.api._base import BaseRequestAPI
from amplo.utils.util import check_dtypes

__all__ = [
    "AmploPlatformAPI",
    "upload_model",
    "report_training_fail",
]


_PLATFORM_HOST = "https://platform.amplo.ch"
_PLATFORM_TOKEN_OS = "AMPLO_PLATFORM_STRING"


def _format_version(version: int | str | None) -> str | None:
    """
    If provided, validates and formats a model version.

    Parameters
    ----------
    version : int or str
        Version to be validated

    Returns
    -------
    str
        If version is not None, returns formatted model version (e.g. "v1", "v2", ...).

    Raises
    ------
    ValueError
        When `version` is malformed, i.e. not mutable to an integer.
    """

    if version is None:
        return None

    # Stringify and append "v" in front of it
    version = re.sub("^v+", "v", f"v{version}")

    # Validity check
    try:
        _ = int(version.removeprefix("v"))
    except ValueError as err:
        raise ValueError(f"Parameter `version` is malformed: {err}")

    return version


class AmploPlatformAPI(BaseRequestAPI):
    """
    Helper class for working with Amplo's Platform API.

    Parameters
    ----------
    host : str
        Amplo platform host.
    access_token : str
        Access token, a.k.a. "X-Api-Key", for platform.
    """

    def _authorization_header(self) -> dict[str, str]:
        return {"X-Api-Key": self.access_token}

    @classmethod
    def from_os_env(
        cls, host: str | None = None, access_token_os: str | None = None
    ) -> Self:
        """
        Instantiate the class using os environment strings.

        Parameters
        ----------
        host : str, default: _PLATFORM_HOST
            Amplo host. Not from os environment!
        access_token_os : str, default: _PLATFORM_TOKEN_OS
            Key in the os environment for the platform access token.

        Returns
        -------
        AmploPlatformAPI

        Raises
        ------
        KeyError
            When a os variable is not set.
        """

        access_token_os = access_token_os or _PLATFORM_TOKEN_OS

        host = host or _PLATFORM_HOST
        access_token = os.environ[access_token_os]
        return cls(host, access_token)

    def list_models(
        self,
        team: str | None = None,
        machine: str | None = None,
        service: str | None = None,
        issue: str | None = None,
        **more_params,
    ) -> list[dict[str, str | None]]:
        params = {
            "team": team,
            "machine": machine,
            "category": service,
            "name": issue,
            **more_params,
        }
        return self.request("get", "models", params=params).json()

    def list_trainings(
        self,
        team: str | None = None,
        machine: str | None = None,
        service: str | None = None,
        issue: str | None = None,
        version: str | int | None = None,
        **more_params,
    ) -> list[dict[str, str]]:
        version = _format_version(version)
        params = {
            "team": team,
            "machine": machine,
            "category": service,
            "model": issue,
            "version": version,
            **more_params,
        }
        return self.request("get", "trainings", params=params).json()

    def upload_training(
        self,
        team: str,
        machine: str,
        training_id: int,
        files: list[tuple[str, BinaryIO]] | None = None,
        status: int | None = None,
    ) -> Response:
        data = {
            "team": team,
            "machine": machine,
            "id": training_id,
            "new_status": status,
        }
        return super().request("put", "trainings", data=data, files=files)

    def get_datalogs(
        self,
        team: str,
        machine: str,
        category: str | None = None,
        **more_params,
    ) -> list[dict[str, str | None]]:
        if more_params.get("filename", False):
            warn(
                "Found 'filename' key in 'more_params'. "
                "Consider using the method 'get_datalog' instead."
            )
        params = {"team": team, "machine": machine, "category": category, **more_params}
        return self.request("get", "datalogs", params=params).json()

    def get_datalog(
        self,
        team: str,
        machine: str,
        filename: str,
    ) -> dict[Any, Any]:
        params = {"team": team, "machine": machine, "filename": filename}
        return self.request("get", "datalogs", params=params).json()


def upload_model(
    train_id: int,
    model_dir: str | Path,
    team: str,
    machine: str,
    service: str,
    issue: str,
    version: str | int,
    *,
    host: str | None = None,
    access_token_os: str | None = None,
) -> Response:
    """
    Uploads a trained model to the Amplo platform.

    Notes
    -----
    Make sure to have set the following environment variables:
        - ``AMPLO_PLATFORM_STRING`` (access token for platform).

    Parameters
    ----------
    model_id : int
        Model training ID.
    model_dir : str or Path
        Model directory which contains a "Production/v{version}/Settings.json" and
        "Production/v{version}/Model.joblib" file.
    team : str
        Name of the team.
    machine : str
        Name of the machine.
    service : str
        Name of the service (a.k.a. category).
    issue : str
        Name of the issue (a.k.a. model).
    version : str or int
        Model version ID, e.g. "v1".
    host : str, default: _PLATFORM_HOST
        Amplo platform host. Not from os environment!
    access_token_os : str, default: _PLATFORM_TOKEN_OS
        Key in the os environment for the platform access token.

    Returns
    -------
    requests.Response

    Raises
    ------
    ValueError
        When no training exists on the platform with the given parameters.
    """

    check_dtypes(
        ("train_id", train_id, int),
        ("model_dir", model_dir, (str, Path)),
        ("team", team, str),
        ("machine", machine, str),
        ("service", service, str),
        ("issue", issue, str),
        ("version", version, (str, int)),
    )

    # Check directory
    model_dir = Path(model_dir) / str(_format_version(version))
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Invalid `model_dir` directory: {model_dir}")

    # Check and set model files
    model_files: list[tuple[str, BinaryIO]] = []
    for file in ("Settings.json", "Model.joblib"):
        if not (model_dir / file).exists():
            raise FileNotFoundError(f"File '{file}' not found in '{model_dir}'.")
        io = open(model_dir / file, "rb")
        model_files.append(("files", io))

    # Check that the training of the model exists
    api = AmploPlatformAPI.from_os_env(host, access_token_os)
    trainings = api.list_trainings(team, machine, service, issue, version)
    if len(trainings) != 1:
        raise ValueError("There exists no training with the given parameters.")

    return api.upload_training(
        team=team, machine=machine, training_id=train_id, files=model_files, status=2
    )


def report_training_fail(
    team: str,
    machine: str,
    training_id: int,
    *,
    host: str | None = None,
    access_token_os: str | None = None,
):
    """
    Report training status "Failed" to the platform.

    Parameters
    ----------
    team : str
        Name of the team.
    machine : str
        Name of the machine.
    training_id : int, optional
        Model training ID.
    host : str, default: _PLATFORM_HOST
        Amplo platform host. Not from os environment!
    access_token_os : str, default: _PLATFORM_TOKEN_OS
        Key in the os environment for the platform access token.
    """

    api = AmploPlatformAPI.from_os_env(host, access_token_os)
    api.upload_training(team, machine, training_id, status=4)  # 4 == "Failed"
