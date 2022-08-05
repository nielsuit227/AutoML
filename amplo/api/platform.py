#  Copyright (c) 2022 by Amplo.

import json
import os
import re
import warnings
from pathlib import Path

import requests

__all__ = ["PlatformSynchronizer"]


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

    def upload_model(self, issue_dir, team, machine, service, issue, version):
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

        Notes
        -----
        Data structures
            - Locally: ``Auto_ML / Production / vX / ...``
            - Cloud: ``Team / Machine / models / Diagnostics / vX / ...``
        """

        # Check input args
        assert os.path.isdir(issue_dir), "Model directory does not exist"
        assert all(
            isinstance(arg, str) for arg in (team, machine, service, issue)
        ), "Invalid argument"
        assert isinstance(version, int), "Invalid argument"

        # Check file structure
        file_dir = Path(issue_dir) / "Production" / f"v{version}"
        assert os.path.isdir(
            file_dir
        ), f"Production files for version {version} do not exist"
        files = [
            file_dir / file for file in ("Settings.json", "Model.joblib", "Report.pdf")
        ]
        for file in files:
            assert file.exists(), f"File {file.name} does not exist"

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
            assert get_response.status_code == 200, "GET request has failed"

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
            assert isinstance(existing_models, list), "Received unexpected data type"
            assert isinstance(existing_models[0], dict), "Received unexpected data type"

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
            assert model_exists(
                data["name"]
            ), f"Failed to create model - {post_response} {post_response.text}"

        # Add files to data
        data.update({"files": [open(file, "r") for file in files]})
        # PUT a new version
        put_response = requests.put(
            self.api_url + "/models", data=data, headers=headers
        )
        assert put_response.status_code == 200

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