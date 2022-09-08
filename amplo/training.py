#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

from amplo import Pipeline
from amplo.api.databricks import DatabricksJobsAPI
from amplo.utils import check_dtypes
from amplo.utils.io import merge_logs

__all__ = ["DEFAULT_PIPE_KWARGS", "train_locally", "train_on_cloud"]

DEFAULT_PIPE_KWARGS = {
    "interval_analyse": False,
    "standardize": False,
    "missing_values": "zero",
    "balance": False,
    "stacking": False,
    "grid_search_time_budget": 7200,
    "n_grid_searches": 1,
    "verbose": 1,
}


def _set_default_pipe_kwargs(
    team: str,
    machine: str,
    service: str,
    issue: str,
    model_version: int,
    unhandled_pipe_kwargs: dict = None,
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


def train_locally(
    data_dir: str | Path,
    target_dir: str | Path,
    team: str,
    machine: str,
    service: str,
    issue: str,
    pipe_kwargs: dict = None,
    model_version: int = 1,
    *,
    working_dir: str | Path = "./tmp",
) -> True:
    """
    Locally train a model with given parameters.

    Parameters
    ----------
    data_dir : str or Path
        Directory where data is stored. Note that it must contain subdirectories which
        names depict the issues (e.g. healthy data).
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
    *
    working_dir : str or Path, default: "./tmp"
        Directory where temporary training files will be stored.
        Note that this directory will be deleted again.

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
    target = pipe_kwargs["target"]
    data, file_metadata = merge_logs(data_dir, target)

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

    # Set up pipeline and train
    pipeline = Pipeline(main_dir=f"{working_dir.as_posix()}/", **pipe_kwargs)
    pipeline.fit(data)

    # Insert file metadata
    pipeline.settings["file_metadata"] = file_metadata

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
    pipe_kwargs: dict = None,
    model_version: int = 1,
    *,
    host_os: str = None,
    access_token_os: str = None,
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
    *
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
        "pipe_kwargs": json.dumps(pipe_kwargs),
    }

    # Send request
    api = DatabricksJobsAPI.from_os_env(host_os, access_token_os)
    return api.run_job(job_id=job_id, notebook_params=notebook_params)
