#  Copyright (c) 2022 by Amplo.

import warnings
from typing import Any
from unittest.mock import DEFAULT

import pytest

from amplo import Pipeline
from amplo.training import DEFAULT_PIPE_KWARGS, _set_default_pipe_kwargs


def test_default_pipe_kwargs():
    with warnings.catch_warnings(record=True) as warns:
        Pipeline(**DEFAULT_PIPE_KWARGS)
    assert not any(
        isinstance(w.message, UserWarning) for w in warns
    ), "Unexpected pipe kwarg"


def test_set_default_pipe_kwargs():
    params: dict[str, Any] = dict(team="1", machine="2", service="3", issue="4")
    params["model_version"] = 1

    # Case 1: no kwargs are given
    pipe_kwargs = _set_default_pipe_kwargs(**params, unhandled_pipe_kwargs=None)
    # Test that default values are set
    assert {key: pipe_kwargs[key] for key in DEFAULT_PIPE_KWARGS} == DEFAULT_PIPE_KWARGS
    # Test that name, target and version are set
    assert pipe_kwargs["name"] == "1 - 2 - 3 - 4"
    assert pipe_kwargs["target"] == "target"
    assert pipe_kwargs["version"] == params["model_version"]

    # Case 2: kwargs are given and shall not be overwritten
    pipe_kwargs = _set_default_pipe_kwargs(
        **params, unhandled_pipe_kwargs={"target": "labels", "verbose": 2}
    )
    # Test that given values came through
    assert pipe_kwargs["target"] == "labels"
    assert pipe_kwargs["verbose"] == 2
    # Test that the other default parameters are not overwritten
    default_pipe_kwargs = DEFAULT_PIPE_KWARGS
    default_pipe_kwargs.pop("verbose")
    assert {key: pipe_kwargs[key] for key in default_pipe_kwargs} == DEFAULT_PIPE_KWARGS


def test_train_on_premise():
    pytest.skip("Not yet implemented.")


def test_train_on_cloud():
    pytest.skip("Not yet implemented.")
