#  Copyright (c) 2022 by Amplo.

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.stats

__all__ = [
    "DummyDataSampler",
    "make_data",
    "make_cat_data",
    "make_num_data",
    "make_interval_data",
    "make_production_data",
]


class DummyDataSampler:
    def __init__(self, distribution=scipy.stats.norm):
        """
        Class for sampling from a given distribution

        Parameters
        ----------
        distribution : scipy.stats.rv_continuous
            Distribution to sample from
        """
        self.transform = distribution

    def sample_data(self, num_samples: int):
        """Sample data"""
        cum = np.random.uniform(0.01, 0.99, num_samples)
        data = self.transform.ppf(cum)
        return data


def make_data(num_samples, *, cat_choices=None, num_dists=None):
    """
    Randomly sample categorical and/or numerical dummy data

    Parameters
    ----------
    num_samples : int
        Number of samples
    cat_choices : list of str or list of list or bool, optional
        (categorical) Specifies the choices to sample from
    num_dists : str or list of str or bool, optional
        (numerical) Specifies the distributions to sample from

    Returns
    -------
    tuple (pd.DataFrame, dict of {str : list})
    """

    # Handle input
    def parse_input(input_, default_value, list_of_list=False, list_of_str=False):
        if input_ is None or input_ is True:
            return default_value
        elif not input_:
            return []  # leave it empty
        elif (list_of_list and not isinstance(input_[0], list)) or (
            list_of_str and not isinstance(input_, list)
        ):
            return [input_]
        return input_

    cat_choices = parse_input(
        cat_choices, [list("abc"), list("xyz")], list_of_list=True
    )
    num_dists = parse_input(
        num_dists,
        ["uniform::0::100", "norm", "expon::0.4", "gamma::0.3", "beta::0.2::0.4"],
        list_of_str=True,
    )
    assert any(
        [cat_choices, num_dists]
    ), "Please specify at least one categorical or numerical column"

    # Sample categorical data
    cat_df = pd.DataFrame()
    for i, choice in enumerate(cat_choices):
        cat_df[f"cat_{i}"] = np.random.choice(choice, (num_samples,))

    # Sample numerical data
    num_df = pd.DataFrame()
    for i, dist in enumerate(num_dists):
        # Convert string to function
        if isinstance(dist, str):
            splits = dist.split("::")
            dist = splits[0]
            args = [float(arg) for arg in splits[1:]]
            dist = getattr(scipy.stats, dist)(*args)
        # Add distribution to corresponding array
        num_df[f"num_{i}"] = DummyDataSampler(dist).sample_data(num_samples)

    # Concatenate all data (which is not empty)
    df = pd.concat([df_ for df_ in (cat_df, num_df) if not df_.empty], axis=1)
    info = dict(cat_cols=cat_df.columns, num_cols=num_df.columns)

    return df, info


def make_cat_data(num_samples: int, cat_choices=None):
    return make_data(num_samples, cat_choices=cat_choices, num_dists=False)


def make_num_data(num_samples: int, num_dists=None):
    return make_data(num_samples, cat_choices=False, num_dists=num_dists)


def make_interval_data(n_logs=2, n_labels=2, directory=None, target="labels", **kwargs):
    """
    Create dummy data for Amplo`s IntervalAnalyser

    Parameters
    ----------
    n_logs : int
        Number of dummy logs / log files to create
    n_labels : int
        Number of dummy labels per log file
    directory : str or Path, optional
        Parent directory where interval data will be stored.
        If no directory is provided it will return a pandas.DataFrame instead.
    target : str
        Name of target column
    **kwargs
        Keyword arguments will be passed to ``make_data`` function

    Returns
    -------
    pd.DataFrame, optional
        Multi-indexed DataFrame such as the IntervalAnalyser would return it
    """

    # Define log names
    log_names = [f"DummyLog_{i}" for i in range(n_logs)]
    label_names = [f"DummyLabel_{i}" for i in range(n_labels)]
    # List of dataframes
    dfs = []

    for log in log_names:
        for label in label_names:

            num_samples = 20
            x, _ = make_data(num_samples, **kwargs)

            if directory:
                # Save in folder / label / log.csv
                save_dir = Path(directory) / label
                save_dir.mkdir(parents=True, exist_ok=True)
                x.to_csv(save_dir / f"{log}.csv", index=False)
            else:
                # Set multi-index
                index = pd.MultiIndex.from_arrays(
                    [num_samples * [log], range(num_samples)], names=["log", "index"]
                )
                x.set_index(index, inplace=True)
                # Add label column
                x[target] = label
                # Append to list
                dfs += [x]

    if not directory:
        return pd.concat(dfs)


def make_production_data(
    main_dir,
    team="DummyTeam",
    machine="DummyMachine",
    service="DummyService",
    issue="DummyIssue",
    version=1,
):
    """
    Create dummy production data, simulating output of the Amplo`s Pipeline

    Parameters
    ----------
    main_dir : str or Path
    team : str
    machine : str
    service : str
    issue : str
    version : int

    Returns
    -------
    tuple of Path, dict
        Model issue path AND dictionary containing info about dummy model
    """
    # Setup
    issue_dir = Path(main_dir) / team / machine / service / "models" / issue
    production_v_dir = issue_dir / "Production" / f"v{version}"
    production_v_dir.mkdir(parents=True, exist_ok=True)

    # Save dummy settings
    json.dump(dict(), open(production_v_dir / "Settings.json", "w"))
    # Save dummy model
    joblib.dump("", production_v_dir / "Model.joblib")
    # Save dummy report
    (production_v_dir / "Report.pdf").touch()

    return issue_dir, dict(
        team=team, machine=machine, service=service, issue=issue, version=version
    )
