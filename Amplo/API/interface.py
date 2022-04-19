import re
import time
import warnings
from pathlib import Path

from termcolor import cprint
import pandas as pd

from Amplo import Pipeline
from Amplo.AutoML import IntervalAnalyser
from Amplo.API.storage import AzureSynchronizer
from Amplo.API.platform import PlatformSynchronizer


__all__ = ['API']


class API:

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
            verbose=0,
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

        if local_data_dir is None:
            warnings.warn('No local data directory provided. Falling back to using current working directory.')
            local_data_dir = '.'

        self.dataDir = Path(local_data_dir)
        self._download_data = bool(download_data)
        self._train_model = True
        self._upload_model = bool(upload_model)
        self._trained_model_args = list()  # arguments for uploading model

        # Set up storage
        self.storage = AzureSynchronizer()
        self.platform = PlatformSynchronizer()

        # Set data arguments
        self.teams = self._set_azure_arg(teams)
        self.machines = self._set_azure_arg(machines)
        self.services = self._set_azure_arg(services)
        self.issues = self._set_azure_arg(issues)

        # TODO: Use Amplo`s logging instead of a print function
        self.verbose = int(verbose)

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
            return ['*']
        else:
            raise ValueError(f'List, string or None-type expected but got {type(arg)} instead.')

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

        self.print('Downloading new data...\n', pre='\n')

        def blob_paths(curr_path=None, selection=('*',)):
            """
            Get all blob paths

            Parameters
            ----------
            curr_path : str or None
            selection : list of str

            Returns
            -------
            list of str
                All paths that match the selection
            """
            all_paths = self.storage.get_dir_paths(curr_path)
            if '*' in selection:
                return all_paths
            return [p for p in all_paths if Path(p).name in selection]

        # For every team
        for team_dir in blob_paths(None, self.teams):
            # For every machine
            for mc, machine_dir in enumerate(blob_paths(team_dir, self.machines)):
                # For every service
                for sc, service_dir in enumerate(blob_paths(f'{machine_dir}/data/', self.services)):
                    # For every issue
                    for ic, issue_dir in enumerate(blob_paths(service_dir, self.issues)):

                        # Get names
                        team = Path(team_dir).name
                        machine = Path(machine_dir).name
                        service = Path(service_dir).name
                        issue = Path(issue_dir).name

                        # Logging
                        if self.verbose > 0 and all(count == 0 for count in [mc, sc, ic][:self.verbose]):
                            # Depending on the verbosity, print only part of all logging info.
                            #  Furthermore, do not print several times the same info.
                            log_info = [f'Team: {team}', f'Machine: {machine}',
                                        f'Service: {service}', f'Issue: {issue}']
                            self.print('Download -- ' + ', '.join(log_info[:self.verbose]))

                        # Define local saving directory (note the switch of `data`s position)
                        local_issue_dir = str(self.dataDir / team / machine / service / 'data' / issue)
                        # Clean spaces in path
                        local_issue_dir = re.sub(r'\s+', ' ', local_issue_dir)  # double spaces to single spaces
                        local_issue_dir = re.sub(r'^\s+', '', local_issue_dir)  # remove preceding spaces
                        local_issue_dir = re.sub(r'\s+$', '', local_issue_dir)  # remove subsequent spaces

                        # Synchronize all files of given issue
                        self.storage.sync_files(issue_dir, local_issue_dir)

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

        self.print('Start model training...\n', pre='\n')

        for data_path, model_info in self._iter_data():
            team = model_info['team']
            machine = model_info['machine']
            service = model_info['service']

            self.print(f'Training models for: {team} / {machine} / {service} \t(team/machine/service)\n', pre='\n')

            # Set up directories
            read_dir = data_path / 'data'
            save_dir = data_path / 'data_IA'
            model_dir = data_path / 'models'
            save_dir.mkdir(exist_ok=True)
            model_dir.mkdir(exist_ok=True)

            # Interval-analyze data
            data = IntervalAnalyser(folder=str(read_dir)).fit_transform()
            # Read the latest data, if any
            latest_data = self.read_latest_version(save_dir)

            # Skip training model when no new data
            if data.equals(latest_data):
                self.print(f'Skipped training for machine {machine} as no new data was found.', 'gray')
                continue

            # Store for versioning and continue to train model
            data.to_csv(save_dir / f'training_data_{int(time.time())}.csv')

            # Select issues to iterate
            all_issues = data['labels'].unique()
            iter_issues = all_issues if '*' in self.issues else self.issues

            # Train one model for each issue
            for issue in iter_issues:
                issue_dir = f'{model_dir}/{issue}/'
                issue_name = f'{team} - {machine} - {service} - {issue}'

                self.print(f'Training model: {issue_name}', 'blue', pre='\n')
                # Prepare data
                data_copy = data.copy()
                data_copy['labels'] = pd.Series(data['labels'] == issue, dtype='int')
                # Define pipeline arguments
                pipe_kwargs = dict(
                    main_dir=issue_dir,
                    target='labels',
                    name=issue_name,
                    extract_features=False,
                    standardize=False,
                    balance=False,
                    grid_search_time_budget=7200,
                    grid_search_candidates=2,
                )
                pipe_kwargs.update(kwargs)  # (optional) manipulation of Pipeline's kwargs
                # Create and fit pipeline
                pipe = Pipeline(**pipe_kwargs)
                pipe.fit(data_copy)

                # Append to trained model arguments
                self._trained_model_args.append([issue_dir, team, machine, service, issue, pipe.version])

    def upload_models(self, model_args=None):
        """
        Upload trained models or, when `model_args` provided,
        upload given models

        Parameters
        ----------
        model_args : list, optional
            Model arguments. List of tuples, whereas each tuple contains following info in order:
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

        self.print('Uploading trained models...\n', pre='\n')

        if model_args is not None and len(self._trained_model_args) > 0:
            warnings.warn(UserWarning('An explicit list of model_args was provided despite > 1 model was trained.'))

        for model_args in model_args or self._trained_model_args:
            # Upload models to Amplo`s platform
            self.platform.upload_model(*model_args)

    # --- Utilities ---

    def print(self, text, fmt='yellow', pre=None):
        """
        Custom printer function.

        Parameters
        ----------
        text : str
            Text to print
        fmt : str
            Formatting info for ``termcolor.cprint``.
        pre : str, optional
            String to attach at the very beginning
        """

        if self.verbose:
            pre = str(pre) if pre is not None else ''
            cprint(pre + '[AmploAPI] ' + text, fmt)

    def _iter_data(self):
        """
        Iterate over all services that are specified by `self.teams`, `self.machines` and `self.services`
        and follow the following directory structure:
            ``data_dir / team_dir / machine_dir / service_dir``

        Yields
        ------
        (Path, dict of {str: str})
            Tuple of all matches and some info aside.

            Note that the path_info is somehow redundant, as it's recoverable by the data path.
            Out of convenience, however, we'll keep that info at the side.
        """

        def iterate_dirs(dir_, match_list):
            """
            Iterate directory and yield all subdirectories that match with one of the
            RegEx expressions in the match list.

            Parameters
            ----------
            dir_ : str or Path
                Directory to search for.
            match_list : list of str
                List of RexEx expressions to match directories.

            Yields
            ------
            Path
            """
            for match in match_list:
                for sub_dir in Path(dir_).glob(match):
                    if not sub_dir.is_dir():
                        continue
                    yield sub_dir

        for team_dir in iterate_dirs(self.dataDir, self.teams):
            for machine_dir in iterate_dirs(team_dir, self.machines):
                for service_dir in iterate_dirs(machine_dir, self.services):
                    data_path = service_dir.resolve()
                    path_info = dict(team=team_dir.name, machine=machine_dir.name, service=service_dir.name)
                    yield data_path, path_info

    @staticmethod
    def read_latest_version(data_dir):
        """
        Read out the latest version of multi-indexed *.csv data and remove unnamed columns.

        Parameters
        ----------
        data_dir : str or Path
            directory to search *.csv files

        Returns
        -------
        pd.DataFrame or None
            The latest version of data, or None if no version was found
        """

        # Find path to the latest data version
        all_versions = list(Path(data_dir).glob('*.csv'))
        if len(all_versions) == 0:
            # There is nothing to read
            return None

        # Read data
        path_to_latest = max(all_versions)
        data = pd.read_csv(path_to_latest, index_col=[0, 1])

        return data
