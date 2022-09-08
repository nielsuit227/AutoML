#  Copyright (c) 2022 by Amplo.

import re
import warnings
from copy import deepcopy
from pathlib import Path

import pandas as pd

from amplo.api.platform import PlatformSynchronizer
from amplo.api.storage import AzureSynchronizer
from amplo.base.objects import LoggingMixin
from amplo.pipeline import Pipeline
from amplo.utils import deprecated

__all__ = ["API"]


@deprecated("This class won't be updated anymore.")
class API(LoggingMixin):
    def __init__(
        self,
        local_data_dir=None,
        download_data=True,
        upload_model=True,
        *,
        teams=None,
        machines=None,
        services=None,
        issues=None,
        verbose=1,
    ):
        """
        API for downloading data from blob storage, training models with AutoML and
        uploading them to Amplo`s platform.

        Parameters
        ----------
        local_data_dir : str or Path
            Destination directory for synchronizing data.
        download_data : bool
            Whether to synchronize from Azure blob storage.
        upload_model : bool
            Whether to upload trained model to platform.

        teams : list of str, optional
            Specify which teams will be considered for processing.
        machines : list of str, optional
            Specify which machines will be considered for processing.
        services : list of str, optional
            Specify which services will be considered for processing.
        issues : list of str, optional
            Specify which issues will be considered for processing.

        verbose : int
            Logging verbosity
        """
        super().__init__(verbose=verbose)

        if local_data_dir is None:
            warnings.warn(
                "No local data directory provided. Falling back to using current "
                "working directory."
            )
            local_data_dir = "."

        self.data_dir = Path(local_data_dir)
        self._download_data = bool(download_data)
        self._train_model = True
        self._upload_model = bool(upload_model)

        # Internal info
        self._trained_model_args = list()  # arguments for uploading model

        # Set up storage
        self.storage = AzureSynchronizer()
        self.platform = PlatformSynchronizer()

        # Set data arguments
        self.teams = self._set_azure_arg(teams)
        self.machines = self._set_azure_arg(machines)
        self.services = self._set_azure_arg(services)
        self.issues = self._set_azure_arg(issues)

    @staticmethod
    def _set_azure_arg(arg):
        """
        Put Azure argument into a list

        Parameters
        ----------
        arg : list or str or None

        Returns
        -------
        list of str
        """
        if isinstance(arg, list):
            return arg
        elif isinstance(arg, str):
            return [arg]
        elif arg is None:
            return ["*"]
        else:
            raise ValueError(
                f"List, string or None-type expected but got {type(arg)} instead."
            )

    def fit(self):
        self.sync_data()
        self.train_models()
        self.upload_models()

    def sync_data(self):
        """
        Synchronizes data from Azure blob storage

        Notes
        -----
        The algorithm iterates through all valid paths according to `self.teams`,
        `self.machines` and `self.services`. Note, however, that always all issues
        will be downloaded and cannot be specifically targeted
        """
        if not self._download_data:
            return

        self.logger.info("Downloading new data...")

        # Iterate blob data paths
        for issue_dir, team, machine, service, issue in self._iterate_data(
            from_local=False, level="issue"
        ):

            # Logging
            if self.verbose > 0:
                self.logger.info(
                    f"Downloading from: [Team] {team} - [Machine] {machine} - "
                    f"[Service] {service} - [Issue] {issue}"
                )

            # Define local saving directory (note the switch of `data`s position)
            local_issue_dir = str(
                self.data_dir / team / machine / service / "data" / issue
            )
            # Clean spaces in path
            local_issue_dir = re.sub(
                r"\s+", " ", local_issue_dir
            )  # double spaces to single spaces
            local_issue_dir = re.sub(
                r"^\s+", "", local_issue_dir
            )  # remove preceding spaces
            local_issue_dir = re.sub(
                r"\s+$", "", local_issue_dir
            )  # remove subsequent spaces

            # Synchronize all files of given issue
            self.storage.sync_files(
                issue_dir, local_issue_dir
            )  # TODO give `last_updated` argument

    def train_models(self, **kwargs):
        """
        Trains a model for every combination of
            a) team
            b) machine
            c) service
            d) issue

        Note that for the first 3 a selection can be specified in `self.__init__`.

        Parameters
        ----------
        **kwargs : optional
            Arguments to manipulate behavior of `Amplo.Pipeline`
        """

        if not self._train_model:
            return

        self.logger.info("Start model training...")

        for service_dir, team, machine, service in self._iterate_data(
            from_local=True, level="service"
        ):

            self.logger.info(
                f"Preparing data for: "
                f"[Team] {team} - [Machine] {machine} - [Service] {service}",
            )

            # Set up directories
            read_dir = service_dir / "data"
            model_dir = service_dir / "models"
            model_dir.mkdir(exist_ok=True)

            # --- Prepare common data (all models will act on the essentially same data)

            # Set pipeline arguments
            preparing_pipeline_kwargs = dict(
                target="labels",
                extract_features=False,
                balance=False,
                grid_search_timeout=7200,
                n_grid_searches=2,
            )
            preparing_pipeline_kwargs.update(kwargs)  # allow (optional) manipulation
            preparing_pipeline_kwargs["main_dir"] = f"{model_dir}/_Data_Preparation/"
            preparing_pipeline_kwargs[
                "name"
            ] = f"Preparing Pipeline ({team} - {machine} - {service})"

            # Check whether new data has been added
            checker = Pipeline(no_dirs=True, **preparing_pipeline_kwargs)
            if Path(checker.main_dir).exists():
                checker._load_version()  # noqa
            else:
                checker.version = 1
            # TODO: load settings directly, if present
            checker._read_data(read_dir)  # noqa
            if not checker.has_new_training_data():
                self.logger.info("Skipped training since no new data")
                continue

            # Set up pipeline and prepare data
            preparing_pipeline = Pipeline(**preparing_pipeline_kwargs)
            preparing_pipeline.data_preparation(read_dir)

            # Select issues to iterate
            all_issues = preparing_pipeline.y_orig.unique()
            iter_issues = all_issues if "*" in self.issues else self.issues

            # Train one model for each issue
            for issue in iter_issues:
                self.logger.info(f"Training model for {issue}")

                # Set up pipeline using a copy from the preparing pipeline
                pipeline = deepcopy(preparing_pipeline)
                pipeline.main_dir = f"{model_dir}/{issue}/"
                pipeline.name = f"{team} - {machine} - {service} - {issue}"

                # Create dirs
                if not pipeline.no_dirs:
                    pipeline._create_dirs()  # noqa
                    pipeline._load_version()  # noqa

                # Inject model specific labels
                new_y = pd.Series(pipeline.y_orig == issue, dtype="int32")
                pipeline.set_y(new_y)
                pipeline.n_classes = 2

                # Train models
                pipeline.model_training()
                pipeline.conclude_fitting()

                # Append to trained model arguments
                self._trained_model_args.append(
                    [pipeline.main_dir, team, machine, service, issue, pipeline.version]
                )

    def upload_models(self, model_args=None):
        """
        Upload trained models or, when `model_args` provided, upload given models.

        Parameters
        ----------
        model_args : typing.List[list or tuple], optional
            Model arguments. Each tuple contains following info in order:
                - issue directory [str or Path]
                - team [str]
                - machine [str]
                - service [str]
                - issue [str]
                - version [int]
        """

        if not self._upload_model and not model_args:
            # Skip if setting says not to upload trained models
            #  and no explicit model_args have been passed.
            return

        self.logger.info("Uploading trained models...")

        if model_args is not None and len(self._trained_model_args) > 0:
            warnings.warn(
                "An explicit list of model_args was provided despite > 1 model was "
                "trained.",
                UserWarning,
            )

        for model_args in model_args or self._trained_model_args:
            # Upload models to Amplo`s platform
            self.platform.upload_model(*model_args)

    # --- Utilities ---

    def _iterate_data(self, *, from_local=True, level="issue"):
        """
        Iterate over directories

        Parameters
        ----------
        from_local : bool
            Whether `parent_dir` is a local or a blob directory path
        level : str
            Select level of iterating data.
            One of ('team', 'machine', 'service', 'issue')

        Yields
        ------
        dir : Path
            Directory of data
        *info : str
            Names of team, machine, service and issue.
            Depends on the `level` parameter!
        """

        # Assertions
        assert level in ("team", "machine", "service", "issue"), "Unknown argument"

        # Init
        parent_dir = self.data_dir if from_local else None

        # Iterate teams
        for team_dir in self._iterate_directory(
            parent_dir, self.teams, from_local=from_local
        ):
            info = (team_dir.name,)

            if level == "team":
                yield team_dir, *info
                continue

            # Iterate machines
            for machine_dir in self._iterate_directory(
                team_dir, self.machines, from_local=from_local
            ):
                info = (team_dir.name, machine_dir.name)

                if level == "machine":
                    yield machine_dir, *info
                    continue

                # Adjust search path
                machine_dir_ = (
                    Path(f"{machine_dir}/data/") if not from_local else machine_dir
                )

                # Iterate services
                for service_dir in self._iterate_directory(
                    machine_dir_, self.services, from_local=from_local
                ):
                    info = (team_dir.name, machine_dir.name, service_dir.name)

                    if level == "service":
                        yield service_dir, *info
                        continue

                    # Adjust search path
                    service_dir_ = (
                        Path(f"{service_dir}/data/") if from_local else service_dir
                    )

                    # Iterate issues
                    for issue_dir in self._iterate_directory(
                        service_dir_, self.issues, from_local=from_local
                    ):
                        info = (
                            team_dir.name,
                            machine_dir.name,
                            service_dir.name,
                            issue_dir.name,
                        )

                        assert level == "issue"
                        yield issue_dir, *info

    def _iterate_directory(self, parent_dir=None, selection=None, from_local=True):
        """
        Iterate over subdirectories of parent directory.

        Parameters
        ----------
        parent_dir : str or Path, optional
            Parent directory.
        selection : str or list of str, optional
            Select subdirectories.
        from_local : bool
            Whether `parent_dir` is a local or a blob directory path.

        Notes
        -----
        The parameter `selection` recognizes RegEx flags. For example,
        ``selection="folder.*"`` would yield each subdirectory that begins with
        ``folder`` (``folder1``, ``folder_two``, ...).

        Yields
        ------
        Path
            Next subdirectory.
        """

        # Input checks
        if not isinstance(parent_dir, (str, Path)):
            err = ValueError(f"Invalid argument type: {type(parent_dir)}")
            if from_local:
                raise err
            elif parent_dir is not None:
                raise err
        if from_local and not Path(parent_dir).exists():
            raise FileNotFoundError(f"Directory does not exist: {parent_dir}")
        if selection is not None:
            if not hasattr(selection, "__iter__"):
                raise ValueError(f"Selection is not iterable: {selection}")
            elif not all(isinstance(sel, str) for sel in selection):
                raise ValueError("Selection must be an iterable of strings.")

        # Make selection regex-able
        if isinstance(selection, str):
            selection = [selection]
        if selection is None or any(sel in ("*", ".*") for sel in selection):
            selection = [".*"]
        else:
            selection = [sel + "$" for sel in selection]

        # Define sorted subdirectory iterator
        if from_local:
            subdirectories = sorted(
                child.resolve()
                for child in Path(parent_dir).iterdir()
                if child.is_dir()
            )
        else:
            subdirectories = sorted(
                Path(child) for child in self.storage.get_dir_paths(parent_dir)
            )

        # Yield matching subdirectories
        for subdir in subdirectories:
            # Note that matches for `re.match` must match at the beginning of a
            # given string. Otherwise, `re.find` would have been used.
            if any(re.match(match, subdir.name) for match in selection):
                yield subdir
