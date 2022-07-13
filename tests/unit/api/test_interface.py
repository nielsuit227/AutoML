#  Copyright (c) 2022 by Amplo.

import itertools
import random

import pytest

from amplo.api import API
from amplo.utils.testing import make_interval_data, make_production_data
from tests.unit.api import TestAPI


class TestInterface(TestAPI):
    def test_iterate_directory_local(self):
        api = API(self.sync_dir, verbose=2)

        # Make local dummy folders
        dummy_subdirs = [
            (self.sync_dir / dummy_subdir).resolve()
            for dummy_subdir in ("test1", "test2", "third_folder")
        ]
        for folder in dummy_subdirs:
            (self.sync_dir / folder).mkdir(parents=True)

        # Check yielding `all` subdirectories
        for selection in (None, "*", ".*"):
            all_subdirs = list(
                api._iterate_directory(
                    parent_dir=self.sync_dir, selection=selection, from_local=True
                )
            )
            assert set(dummy_subdirs) == set(all_subdirs), (
                f"Setting ``selection={selection}`` should return all "
                "subdirectories in the dummy folder."
            )

        # Check yielding an explicit selection
        test_subdirs = list(
            api._iterate_directory(
                parent_dir=self.sync_dir, selection="test1", from_local=True
            )
        )
        assert set(i.resolve() for i in self.sync_dir.glob("test1")) == set(
            test_subdirs
        ), "Invalid explicit selection."
        test_subdirs_via_list = list(
            api._iterate_directory(
                parent_dir=self.sync_dir, selection=["test1"], from_local=True
            )
        )
        assert set(test_subdirs_via_list) == set(
            test_subdirs
        ), "Giving either a str or list of str should yield the same results."

        # Check yielding a selection via RegEx
        test_subdirs = list(
            api._iterate_directory(
                parent_dir=self.sync_dir, selection="test.*", from_local=True
            )
        )
        assert set(i.resolve() for i in self.sync_dir.glob("test*")) == set(
            test_subdirs
        ), "Invalid RegEx selection."

    def test_iterate_directory_blob(self):
        api = API(self.sync_dir, verbose=2)

        all_subdirs = list(
            api._iterate_directory(parent_dir="Demo", selection="*", from_local=False)
        )
        test_subdirs = list(
            api._iterate_directory(
                parent_dir="Demo", selection="Charger 150kW.*", from_local=False
            )
        )
        assert len(test_subdirs) > 0, "List of directories should not be empty."
        assert set(test_subdirs).issubset(
            all_subdirs
        ), "`test` directories should be a subset of `all` directories."
        assert set(test_subdirs) != set(
            all_subdirs
        ), "`test` directories should not be equal to `all` directories."

    def test_iter_data(self):
        # Create dummy dataset
        paths_orig, infos_orig = [], []
        rng = random.Random(354687657938)
        for t, m, s in itertools.product(range(3), repeat=3):
            if rng.choice([True, False]):
                # Create "gaps" in data / leave some out
                continue
            team, machine, service = f"Team{t}", f"Machine{m}", f"Service{s}"
            paths_orig += [(self.sync_dir / f"{team}/{machine}/{service}").resolve()]
            infos_orig += [dict(team=team, machine=machine, service=service)]
            make_interval_data(directory=paths_orig[-1] / "data")

        # Define helper function
        def is_selected(orig_lookup: dict, check_lookup: dict, name: str):
            """Return True if a selection match was found"""
            orig = orig_lookup[f"{name}"]
            check = check_lookup[f"{name}s"]
            if check is None or check == "*" or orig in check:
                return True
            else:
                return False

        # Define test cases
        select_kwargs = [
            # Get all combinations
            dict(teams=None, machines=None, services=None),
            # Single selection
            dict(teams="Team0", machines="Machine0", services="Service0"),
            # Mixed selection
            dict(teams=None, machines="Machine0", services=["Service0", "Service1"]),
        ]
        for kwargs in select_kwargs:
            # Get all paths from ``_iter_data``
            paths_actual, infos_actual = [], []
            for path, team, machine, service in API(
                self.sync_dir, **kwargs
            )._iterate_data(level="service"):
                paths_actual += [path]
                infos_actual += [dict(team=team, machine=machine, service=service)]

            # Define subset
            paths_target, infos_target = [], []
            for path, info in zip(paths_orig, infos_orig):
                if any(not is_selected(info, kwargs, key) for key in info.keys()):
                    continue
                paths_target += [path]
                infos_target += [info]

            # Assertions
            assert set(paths_actual) == set(
                paths_target
            ), f"Path check has failed with kwargs: {kwargs}"
            assert set(str(i) for i in infos_actual) == set(
                str(i) for i in infos_target
            ), f"Info check has failed with kwargs: {kwargs}"

    def test_download(self):
        # TODO: Set up a data selection that is quick to download
        # select = dict(teams='Amplo', machines='TestMachine', services='Unlabelled')
        select = dict(
            teams="Demo",
            machines="Charger 75kW",
            services="Diagnostics",
            issues="Blocked Radiator",
        )
        # Set up API and synch data
        api = API(self.sync_dir, **select, verbose=2)
        api.sync_data()
        # Assert existence of folder
        assert (
            self.sync_dir
            / select["teams"]
            / select["machines"]
            / select["services"]
            / "data"
            / select["issues"]
        ).exists(), "Download has failed or is not in assumed directory"

    def test_training(self):
        # Create a small dummy dataset
        dummy_data_path = (
            self.sync_dir / "DummyTeam" / "DummyMachine" / "DummyService" / "data"
        )
        make_interval_data(n_labels=3, directory=dummy_data_path)
        # Set up api and train models
        api = API(self.sync_dir, verbose=2)
        api.train_models(n_grid_searches=0)
        # Assertions
        assert (
            len(api._trained_model_args) > 0
        ), "No model has been trained or failed to append model arguments to list"

    def test_upload(self):
        """
        This only tests that upload_latest_model doesn't error.
            - Good to check whether the file exists in the cloud.
            - Also needs some cleanup to make sure it doesn't stay there.
            - Also does upload_latest_model truly take the latest?
            - What if latest locally and in the cloud are outdated? It needs to go in
              the right folder.
        """
        # TODO: Implement as soon as platform is out of beta mode
        pytest.skip("Test not yet implemented")

        # Make dummy production data
        issue_dir, kwargs = make_production_data(
            self.sync_dir, team="Demo", machine="Charger 75kW", service="Diagnostics"
        )
        # Set up api and upload model
        api = API(self.sync_dir, verbose=2)
        api.upload_models(
            [
                [
                    issue_dir,
                    kwargs["team"],
                    kwargs["machine"],
                    kwargs["service"],
                    kwargs["issue"],
                    kwargs["version"],
                ]
            ]
        )
