#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import requests

from amplo.api._base import BaseRequestAPI
from amplo.utils import check_dtypes, deprecated

if TYPE_CHECKING:
    from requests import Response

__all__ = [
    "AmploPlatformAPI",
    "upload_model",
    "report_training_fail",
    "PlatformSynchronizer",
]


_PLATFORM_HOST = "https://platform.amplo.ch"
_PLATFORM_TOKEN_OS = "AMPLO_PLATFORM_STRING"


def _format_version(version: int | str) -> str | None:
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

    def __init__(self, host: str, access_token: str):
        super().__init__(host, access_token)

    def _authorization_header(self) -> dict:
        return {"X-Api-Key": self.access_token}

    @classmethod
    def from_os_env(
        cls, host: str = None, access_token_os: str = None
    ) -> AmploPlatformAPI:
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
        """

        host = host or _PLATFORM_HOST
        access_token_os = access_token_os or _PLATFORM_TOKEN_OS
        return cls(host, os.getenv(access_token_os))

    def list_models(
        self,
        team: str = None,
        machine: str = None,
        service: str = None,
        issue: str = None,
        **more_params,
    ) -> list[dict]:
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
        team: str = None,
        machine: str = None,
        service: str = None,
        issue: str = None,
        version: str | int = None,
        **more_params,
    ) -> list[dict]:
        version = _format_version(version)
        params = {
            "team": team,
            "machine": machine,
            "category": service,
            "name": issue,
            "version": version,
            **more_params,
        }
        return self.request("get", "trainings", params=params).json()

    def upload_training(
        self,
        team: str,
        machine: str,
        training_id: int,
        files: list[TextIO] = None,
        status: int = None,
    ) -> Response:
        data = {
            "team": team,
            "machine": machine,
            "id": training_id,
            "files": files,
            "new_status": status,
        }
        return super().request("put", "trainings", data=data)


def upload_model(
    model_id: int,
    model_dir: str | Path,
    team: str,
    machine: str,
    service: str,
    issue: str,
    version: str | int,
    *,
    host: str = None,
    access_token_os: str = None,
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
    *
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
        ("model_id", model_id, int),
        ("model_dir", model_dir, (str, Path)),
        ("team", team, str),
        ("machine", machine, str),
        ("service", service, str),
        ("issue", issue, str),
        ("version", version, (str, int)),
    )

    # Set model directory to the requested version
    model_dir = Path(model_dir) / "Production" / _format_version(version)

    # Check directory
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Invalid `model_dir` directory: {model_dir}")

    # Check and set model files
    model_files = []
    for file in ("Settings.json", "Model.joblib"):
        if not (model_dir / file).exists():
            raise FileNotFoundError(f"File '{file}' not found in '{model_dir}'.")
        io = open(model_dir / file, "r")
        model_files.append(io)

    # Check that the training of the model exists
    api = AmploPlatformAPI.from_os_env(host, access_token_os)
    trainings = api.list_trainings(team, machine, service, issue, version, id=model_id)
    if len(trainings) != 1:
        raise ValueError("There exists no training with the given parameters.")

    return api.upload_training(team, machine, model_id, model_files)


def report_training_fail(
    team: str,
    machine: str,
    training_id: int,
    *,
    host: str = None,
    access_token_os: str = None,
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
    *
    host : str, default: _PLATFORM_HOST
        Amplo platform host. Not from os environment!
    access_token_os : str, default: _PLATFORM_TOKEN_OS
        Key in the os environment for the platform access token.
    """

    api = AmploPlatformAPI.from_os_env(host, access_token_os)
    api.upload_training(team, machine, training_id, status=4)  # 4 == "Failed"

    return


@deprecated("Use `upload_model` (or directly `AmploPlatformAPI`) instead.")
class PlatformSynchronizer:
    def __init__(
        self,
        connection_string_name="AMPLO_PLATFORM_STRING",
        api_url="https://platform.amplo.ch/api",
        verbose=0,
    ):
        """
        Connector to Amplo`s platform for uploading trained models

        Parameters
        ----------
        connection_string_name : str
        api_url : str
        verbose : int
        """
        self.api_url = re.sub(r"/$", "", api_url)
        self.api_key = os.getenv(connection_string_name)
        self.verbose = int(verbose)

    def upload_model(self, issue_dir, team, machine, service, issue, version, model_id):
        """
        Upload trained model

        Parameters
        ----------
        issue_dir : str or Path
            Model issue directory
        team : str
            Team name
        machine : str
            Machine name
        service : str
            Service name
        issue : str
            Issue name
        version : int
            Version of production model to upload
        model_id : int
            Model identifier for platform.

        Notes
        -----
        Data structures
            - Locally: ``Auto_ML / Production / vX / ...``
            - Cloud: ``Team / Machine / models / Diagnostics / vX / ...``
        """

        # Check input args
        if not os.path.isdir(issue_dir):
            raise FileNotFoundError("Model directory does not exist.")
        check_dtypes(
            ("team", team, str),
            ("machine", machine, str),
            ("service", service, str),
            ("issue", issue, str),
            ("version", version, int),
        )

        # Check file structure
        file_dir = Path(issue_dir) / "Production" / f"v{version}"
        if not os.path.isdir(file_dir):
            raise FileNotFoundError(
                f"Production files for version {version} don't exist."
            )
        files = [file_dir / file for file in ("Settings.json", "Model.joblib")]
        for file in files:
            if not file.exists():
                raise FileNotFoundError(f"File {file} doesn't exist in {file_dir}.")

        # Set info
        headers = {"X-Api-Key": self.api_key}
        data = {
            "team": team,  # team name
            "machine": machine,  # machine name
            "category": service,  # category
            "model": issue,
            "name": issue,  # model name (for PUT or POST, respectively)
            "version": f"v{version}",  # latest version
        }

        # Assert that model exists
        def model_exists(model_name: str):
            # GET request
            get_response = requests.get(
                self.api_url + "/models", params=data, headers=headers
            )
            if get_response.status_code != 200:
                raise requests.HTTPError(f"{get_response} {get_response.text}")

            # Get all existing models
            try:
                # Convert string to list of dict
                existing_models = json.loads(get_response.text)
                if isinstance(existing_models, str):
                    ValueError("Could not transform response.")
            except json.decoder.JSONDecodeError as err:
                warnings.warn(f"Could not decode GET response - {err}")
                # Set default value of ``existing_models`` to list of (empty) dict
                existing_models = list(dict())

            # Check existing models
            if not isinstance(existing_models, list):
                raise ValueError(f"Unexpected data type: {type(existing_models)}")
            for model in existing_models:
                if not isinstance(model, dict):
                    raise ValueError(f"Unexpected data type: {type(model)}")

            # Check if model exists
            return any(
                model.get("name", None) == model_name for model in existing_models
            )

        # POST when model yet doesn't exist
        if not model_exists(data["name"]):
            warnings.warn(
                f"Model {data['name']} did not yet exist. POST-ing a new model to the "
                f"platform..."
            )
            post_response = requests.post(
                self.api_url + "/models", data=data, headers=headers
            )
            # Assert that now model exists
            if not model_exists(data["name"]):
                raise requests.HTTPError(f"{post_response} {post_response.text}")

        # Add files and model id to data
        data.update({"files": [open(file, "r") for file in files], "id": model_id})
        # PUT a new version
        put_response = requests.put(
            self.api_url + "/trainings", data=data, headers=headers
        )
        if put_response.status_code != 200:
            raise requests.HTTPError(f"{put_response} {put_response.text}")

    def upload_latest_model(self, issue_dir, *args, **kwargs):
        """
        Same as `upload_model` but selects the latest version of production model

        Parameters
        ----------
        issue_dir : str or Path
            Model issue directory
        args
            Arguments to be passed to `upload_model`
        kwargs
            Keyword arguments to be passed to `upload model`
        """
        # Check input args
        assert os.path.isdir(issue_dir), "Model directory does not exist"

        # Check file structure
        all_version_dirs = [
            dir_
            for dir_ in (Path(issue_dir) / "Production").glob("v*")
            if dir_.is_dir()
        ]
        assert len(all_version_dirs) > 0, "No production data found"

        # Take the latest version and remove `v` in e.g. `v1`
        latest_version = int(sorted(all_version_dirs)[-1].name[1:])

        # Set version in kwargs and upload model
        kwargs.update({"version": latest_version})
        return self.upload_model(issue_dir, *args, **kwargs)
