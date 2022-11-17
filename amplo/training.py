#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import cast
from warnings import warn

from azure.core.exceptions import ResourceNotFoundError
from requests import HTTPError

from amplo import Pipeline
from amplo.api.databricks import DatabricksJobsAPI
from amplo.api.platform import AmploPlatformAPI
from amplo.api.storage import AzureBlobDataAPI
from amplo.utils import check_dtypes
from amplo.utils.io import merge_logs

__all__ = ["DEFAULT_PIPE_KWARGS", "train_locally", "train_on_cloud"]

DEFAULT_PIPE_KWARGS = {
    "interval_analyse": False,
    "standardize": False,
    "missing_values": "zero",
    "balance": False,
    "stacking": False,
    "grid_search_timeout": 7200,
    "n_grid_searches": 1,
    "verbose": 1,
}


def _set_default_pipe_kwargs(
    team: str,
    machine: str,
    service: str,
    issue: str,
    model_version: int,
    unhandled_pipe_kwargs: dict | None = None,
) -> dict:
    """
    Inserts default pipeline arguments if not set already.

    Parameters
    ----------
    team : str
        Name of the team.
    machine : str
        Name of the machine.
    service : str
        Name of the service (a.k.a. category).
    issue : str
        Name of the issue (a.k.a. model).
    model_version : int
        Model version.
    unhandled_pipe_kwargs : dict
        Unhandled keyword arguments for pipeline.

    Returns
    -------
    dict
        Pipeline keyword arguments with imputed values.
    """

    if unhandled_pipe_kwargs is None:
        unhandled_pipe_kwargs = {}

    check_dtypes(
        ("team", team, str),
        ("machine", machine, str),
        ("service", service, str),
        ("issue", issue, str),
        ("model_version", model_version, int),
        ("pipe_kwargs", unhandled_pipe_kwargs, dict),
    )

    pipe_kwargs = deepcopy(DEFAULT_PIPE_KWARGS)

    # Insert unhandled pipeline keyword arguments
    pipe_kwargs.update(unhandled_pipe_kwargs)

    # Set params
    pipe_kwargs["name"] = f"{team} - {machine} - {service} - {issue}"
    pipe_kwargs["target"] = pipe_kwargs.get("target", "target")
    pipe_kwargs["version"] = model_version

    return pipe_kwargs


def _get_file_delta(
    metadata: dict[int, dict],
    team: str,
    machine: str,
    service: str,
    issue: str,
    version: int,
    azure: tuple[str, str] | bool,
    platform: tuple[str, str] | bool,
) -> dict[str, list[str]]:

    # Initialize
    platform_api = AmploPlatformAPI.from_os_env(
        *(platform if not isinstance(platform, bool) else tuple())
    )
    blob_api = AzureBlobDataAPI.from_os_env(
        *(azure if not isinstance(azure, bool) else tuple())
    )

    # Get file_metadata of previous training
    try:
        trainings = platform_api.list_trainings(
            team, machine, service, issue, version - 1
        )
    except HTTPError:  # group matching query does not exist
        trainings = []

    if trainings:
        settings_path = (
            f"{team}/{machine}/models/{service}/{issue}/"
            + trainings[0].get("version", "-")
            + "/Settings.json"
        )
        try:
            settings = blob_api.read_json(settings_path)
            settings = cast(dict, settings)  # type hint
        except ResourceNotFoundError:
            settings = {}
    else:
        settings = {}

    prev_metadata = settings.get("file_metadata", {})

    # Compare files from previous to current version
    curr_files = [meta["full_path"] for meta in metadata.values()]
    prev_files = [meta["full_path"] for meta in prev_metadata.values()]

    return {
        "new_files": sorted(set(curr_files) - set(prev_files)),
        "removed_files": sorted(set(prev_files) - set(curr_files)),
    }


def train_locally(
    data_dir: str | Path,
    target_dir: str | Path,
    team: str,
    machine: str,
    service: str,
    issue: str,
    pipe_kwargs: dict | None = None,
    model_version: int = 1,
    *,
    healthy_data_dir: str | Path | bool = True,
    working_dir: str | Path = "./tmp",
    azure: tuple[str, str] | bool = False,
    platform: tuple[str, str] | bool | None = None,
    logging: bool = True,
) -> bool:
    """
    Locally train a model with given parameters.

    Parameters
    ----------
    data_dir : str or Path
        Directory where data is stored. Note that it must contain subdirectories which
        names depict the issues (e.g. pipe error).
    target_dir : str or Path
        Directory where the trained model files will be copied to.
    team : str
        Name of the team.
    machine : str
        Name of the machine.
    service : str
        Name of the service (a.k.a. category).
    issue : str
        Name of the issue (a.k.a. model).
    pipe_kwargs : dict, optional, default: None
        Keyword arguments for pipeline. Note that defaults will be set.
    model_version : int, default: 1
        Model version.
    healthy_data_dir : str or Path or bool, default: True
        Directory where healthy data is stored.
        If False, no healthy data will be used.
        If True, data will be assumed to be in "../Healthy/Healthy" of `data_dir`.
        Otherwise, will try to read from given path.
    working_dir : str or Path, default: "./tmp"
        Directory where temporary training files will be stored.
        Note that this directory will be deleted again.
    azure : (str, str) or bool, default: False
        Use this parameter to indicate that data is in Azure blob storage.
        If False, it is assumed that data origins from local directory.
        If True, the AzureBlobDataAPI is initialized with default OS env variables.
        Otherwise, it will use the tuple to initialize the api.
    platform : (str, str) or bool or None, default: None
        Use this parameter for selecting data according to Amplo's datalogs.
        If None, its value is set to bool(azure).
        If False, no AmploPlatformAPI will be initialized.
        If True, the AmploPlatformAPI is initialized with default OS env variables.
        Otherwise, it will use the tuple to initialize the api.
    logging : bool, default: True
        Whether to show logging info. Currently only applies to `merge_logs`.

    Returns
    -------
    True
    """

    # Input checks
    pipe_kwargs = _set_default_pipe_kwargs(
        team, machine, service, issue, model_version, pipe_kwargs
    )
    check_dtypes(
        ("data_dir", data_dir, (str, Path)),
        ("target_dir", target_dir, (str, Path)),
        ("working_dir", working_dir, (str, Path)),
    )

    # --- Data ---

    # Read data
    target: str = pipe_kwargs["target"]
    more_folders: list[str | Path]
    if not healthy_data_dir:
        more_folders = []
    elif isinstance(healthy_data_dir, (str, Path)):
        more_folders = [healthy_data_dir]
    else:
        more_folders = [Path(data_dir).parent / "Healthy/Healthy"]
    data, file_metadata = merge_logs(
        data_dir,
        target,
        more_folders=more_folders,
        azure=azure,
        platform=platform,
        logging=logging,
    )

    # Set data labels
    mask = data.loc[:, target] == issue
    data.loc[~mask, target] = 0
    data.loc[mask, target] = 1

    # Safety check
    if data[target].nunique() != 2:
        raise ValueError(f"Number of unique labels is {data[target].nunique()} != 2.")

    # --- Training ---

    # Create temporary working directory
    working_dir = (
        Path(working_dir) / f"{team}_{machine}_{service}_{issue}_{model_version}"
    )
    working_dir.mkdir(parents=True, exist_ok=False)

    # Set up pipeline
    pipeline = Pipeline(main_dir=f"{working_dir.as_posix()}/", **pipe_kwargs)

    # Mirror azure parameter when platform is not set
    if platform is None:
        platform = bool(azure)
    # Indicate new files compared to previous model version
    if azure and platform:
        pipeline.settings["file_delta"] = _get_file_delta(
            file_metadata, team, machine, service, issue, model_version, azure, platform
        )

    # Train
    pipeline.fit(data, metadata=file_metadata)

    # --- Post training ---

    # Move training files into target directory
    shutil.move(working_dir / "Production" / f"v{pipeline.version}", target_dir)

    # Delete temporary working directory
    shutil.rmtree(working_dir)

    return True


def train_on_cloud(
    job_id: int,
    model_id: int,
    team: str,
    machine: str,
    service: str,
    issue: str,
    pipe_kwargs: dict | None = None,
    model_version: int = 1,
    *,
    train_id: int | None = None,
    host_os: str | None = None,
    access_token_os: str | None = None,
) -> dict[str, int]:
    """
    Train a model with given parameters on the cloud (Databricks).

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
    model_id : int
        Model training ID in Amplo's platform.
    team : str
        Name of the team.
    machine : str
        Name of the machine.
    service : str
        Name of the service (a.k.a. category).
    issue : str
        Name of the issue (a.k.a. model).
    pipe_kwargs : dict, optional, default: None
        Keyword arguments for pipeline. Note that defaults will be set.
    model_version : int, default: 1
        Model version.
    train_id : int, optional for now
        Training ID, used for error handling. Will be a required parameter in future editions.
    host_os : str, optional, default: None
        Key in the os environment for the Databricks host.
    access_token_os : str, optional, default: None
        Key in the os environment for the Databricks access token.

    Returns
    -------
    dict of {str: int}
        If response is success (200), ``run_id`` (globally unique key of newly triggered
        run) is one of the present keys.
    """
    if not train_id:
        warn("train_id will be a required parameter in the future.", DeprecationWarning)

    # Input checks
    pipe_kwargs = _set_default_pipe_kwargs(
        team, machine, service, issue, model_version, pipe_kwargs
    )
    check_dtypes(
        ("model_id", model_id, int),
        ("job_id", job_id, int),
        ("host_os", host_os, (type(None), str)),
        ("access_token_os", access_token_os, (type(None), str)),
    )

    # Set up notebook params
    notebook_params = {
        "team": team,
        "machine": machine,
        "service": service,
        "issue": issue,
        "model_id": model_id,
        "train_id": train_id,
        "pipe_kwargs": json.dumps(pipe_kwargs),
    }

    # Send request
    api = DatabricksJobsAPI.from_os_env(host_os, access_token_os)
    return api.run_job(job_id=job_id, notebook_params=notebook_params)
