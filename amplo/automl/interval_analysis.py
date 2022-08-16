#  Copyright (c) 2022 by Amplo.

import warnings
from pathlib import Path
from typing import Union

import faiss
import numpy as np
import pandas as pd

from amplo.automl import DataProcessor
from amplo.automl.feature_processing import FeatureProcessor
from amplo.utils.data import check_dataframe_quality, check_pearson_correlation
from amplo.utils.io import merge_logs
from amplo.utils.logging import logger


class IntervalAnalyser:

    noise = -1

    def __init__(
        self,
        target: str = None,
        norm: str = "euclidean",  # TODO: implement functionality
        min_length: int = 1000,
        n_neighbors: int = None,
        n_trees: int = 10,
        verbose: int = 1,
    ):
        """
        Interval Analyser for Log file classification. Has two purposes:
        - Remove healthy data in longer, faulty logs
        - Remove redundant data in large datasets

        Uses Facebook's FAISS for K-Nearest Neighbors approximation.

        ** IMPORTANT **
        To use this interval analyser, make sure that your logs are located in a
        folder of their label, with one parent folder with all labels, e.g.:
        +-- Parent Folder
        |   +-- Label_1
        |       +-- Log_1.*
        |       +-- Log_2.*
        |   +-- Label_2
        |       +-- Log_3.*

        Parameters
        ----------
        target : str
            Target column name
        norm : str
            Optimization metric for K-Nearest Neighbors
            NOTE: This option has no effect, yet!
        min_length : int
            Minimum length to cut off, everything shorter is left untouched
        n_neighbors : int, optional
            Quantity of neighbors, default to 3 * log length
        n_trees : int
            Quantity of trees
        """
        # Test
        self.available_norms = ["euclidean", "manhattan", "angular", "hamming", "dot"]
        if norm not in self.available_norms:
            raise ValueError(f"Unknown norm, pick from {self.available_norms}")
        # Parameters
        self.target = target or "labels"
        self.min_length = min_length
        self.norm = norm
        self.n_trees = n_trees
        self.n_neighbors = n_neighbors
        self.verbose = verbose

        # Initializers
        self._labels = None
        self._features = None
        self._metadata = None  # optional metadata from `merge_logs`
        # self._mins = pd.DataFrame(index=[0])
        # self._maxs = pd.DataFrame(index=[0])
        self._engine = None
        self._distributions = None
        self._noise_indices = None
        self.avg_samples = None
        self.n_samples = None
        self.n_columns = None
        self.n_files = None
        self.n_folders = None

        # Flags
        self.is_fitted = False

    @property
    def orig_data(self):
        """
        Returns
        -------
        pd.DataFrame
            Original data containing both, feature and label columns
        """
        return pd.concat([self._features, self._labels], axis=1)

    @property
    def data_with_noise(self):
        """
        Returns
        -------
        pd.DataFrame
            Original data where noise is labelled as ``self.noise``

        Notes
        -----
        Depending on the dtype of the noise attribute, the dtype of the labels' column
        will be affected.
        """
        data = self.orig_data
        data.loc[self._noise_indices, self.target] = self.noise
        return data

    @property
    def data_without_noise(self):
        """
        Returns
        -------
        pd.DataFrame
            Original data but without noise rows
        """
        return self.orig_data.drop(self._noise_indices)

    def fit_transform(
        self, data_or_path: Union[pd.DataFrame, str, Path], labels: pd.Series = None
    ):
        """
        Function that runs the K-Nearest Neighbors and returns a dataframe with the
        sensitivities.

        Parameters
        ----------
        data_or_path : pd.DataFrame or str or Path
            Multi-indexed dataset containing feature (and target) columns
            or path to folder that is correctly structured (see Notes)
        labels : pd.Series, optional
            Multi-indexed dataset containing target columns.

        Notes
        -----
        In case you provide a path-like argument to ``features_or_path``,
        make sure that each protocol is located in a sub folder whose name represents
        the respective label.

        A directory structure example:
            |   ``path_to_folder``
            |   ``├─ Label_1``
            |   ``│   ├─ Log_1.*``
            |   ``│   └─ Log_2.*``
            |   ``├─ Label_2``
            |   ``│   └─ Log_3.*``
            |   ``└─ ...``

        Returns
        -------
        pd.DataFrame
            Estimation of correlation strength
        """

        # Parse data
        features, labels = self._parse_data(data_or_path, labels)

        # Set up Annoy Engine (only now that n_keys is known)
        self._build_engine(features)

        # Make distribution
        self._make_distribution(features, labels)

        # Filter dataset
        self._set_noise_indices()

        # Set flags
        self.is_fitted = True

        return self.data_with_noise

    def _parse_data(
        self, data_or_path: Union[pd.DataFrame, str, Path], labels: pd.Series = None
    ):
        """
        Parse data for interval analysis and store it internally

        Parameters
        ----------
        data_or_path : pd.DataFrame or str or Path
            Multi-indexed dataset containing feature (and target) columns
            or path to folder that is correctly structured (see Notes)
        labels : pd.Series, optional
            Multi-indexed dataset containing target columns
        """
        # Verbose
        if self.verbose > 0:
            logger.info("Parsing data for interval analyser.")

        # Parse data
        if isinstance(data_or_path, (str, Path)):
            # Read from path
            features, metadata = merge_logs(data_or_path, self.target)
            labels = features.pop(self.target)
            # Store metadata in self
            self._metadata = metadata
        else:
            if not isinstance(data_or_path, pd.DataFrame):
                raise ValueError("Invalid data_or_path type.")
            # Check if target in `data_or_path`
            if self.target in data_or_path.columns:
                features = data_or_path.drop(self.target, axis=1)
                labels = data_or_path.loc[:, self.target]
            else:
                if not isinstance(labels, pd.Series):
                    raise ValueError("Invalid labels data type.")
                features = data_or_path

        # Tests
        if list(features.index.names) != ["log", "index"]:
            raise ValueError("Invalid multi-indexed data detected.")
        if not all(features.index == labels.index):
            raise ValueError(
                "Indices mismatch: Features and labels cannot be concatenated."
            )
        if len(features) != len(labels):
            raise ValueError(
                "Length mismatch: Features and labels cannot be concatenated."
            )
        if self.target in features.columns:
            raise ValueError("Target column is present in feature data.")

        # Check data quality
        if not check_dataframe_quality(features):
            # Warn & clean
            warnings.warn("Data quality is insufficient, starting DataProcessor.")
            features = DataProcessor().fit_transform(features)

        # Check collinearity
        if not check_pearson_correlation(features, labels):
            warnings.warn("Data is correlated, starting FeatureProcessor.")
            # Remove collinear features and pick Random Forest Increment
            fp = FeatureProcessor(mode="classification", extract_features=False)
            fp.fit(features, labels)
            features = fp.transform(features, feature_set="rf_increment")

        # Set name of labels
        if self.target != labels.name:
            warnings.warn(
                "Expected target name does not match the actual name. "
                f"Name will be fixed from {labels.name} to {self.target}."
            )
            labels.name = self.target

        # Remove datetime columns
        _date_cols = [
            col
            for col in features.columns
            if pd.api.types.is_datetime64_any_dtype(features[col])
        ]
        if len(_date_cols) != 0:
            features.drop(_date_cols, axis=1, inplace=True)

        # Normalize
        self._mins, self._maxs = features.min(), features.max()
        features = (features - features.min()) / (features.max() - features.min())

        # Assert that all dtypes are of dtype float32
        #  See: https://github.com/facebookresearch/faiss/issues/461
        for col in features.select_dtypes(exclude=["float32"]).columns:
            features[col] = features[col].astype("float32")

        # Store data in self
        self._features = features
        self._labels = labels

        # Set sizes
        self.n_folders = labels.nunique()
        self.n_files = features.index.get_level_values(0).nunique()
        self.n_samples, self.n_columns = features.shape
        if self.n_neighbors is None:
            self.n_neighbors = min(3 * self.n_samples // self.n_files, 5000)

        return features, labels

    def _build_engine(self, features: pd.DataFrame):
        """
        Builds the ANNOY engine.

        Parameters
        ----------
        features : pd.DataFrame
            Multi-indexed dataset containing feature columns
        """
        # Create engine
        if self.verbose > 0:
            logger.info("Building interval analyser engine.")
        engine = faiss.IndexFlatL2(self.n_columns)

        # Add the data to ANNOY
        engine.add(np.ascontiguousarray(features.values))  # noqa

        # Set class attribute
        self._engine = engine

    def _make_distribution(self, features: pd.DataFrame, labels: pd.Series):
        """
        Given a build K-Nearest Neighbors, returns the label distribution

        Parameters
        ----------
        features : pd.DataFrame
            Multi-indexed dataset containing feature columns
        labels : pd.Series
            Multi-indexed dataset containing target columns
        """
        if self.verbose > 0:
            logger.info("Calculating interval within-class distributions.")

        # Search nearest neighbors for all samples - has to be iterative for large files
        distribution = []
        for i, row in features.iterrows():
            _, neighbors = self._engine.search(
                np.ascontiguousarray(row.values.reshape((1, -1))), int(self.n_neighbors)
            )
            match_mask = labels.iloc[neighbors.reshape(-1)] == labels.loc[i]
            distribution.append(pd.Series(match_mask).sum() / self.n_neighbors)

        # Parse into list of lists
        self._distributions = pd.Series(distribution, index=self._labels.index)

    def _set_noise_indices(self):
        """
        This function selects samples given the calculated distributions. It only
        removes samples from logs which are longer (> min_length), and only the samples
        with lower in-class neighbors.

        One could come up with a fancier logic, using the total dataset samples, the
        class-balance & sample redundancy.
        """
        # Verbose
        if self.verbose > 0:
            logger.info("Creating filtered dataset")

        # Get in-class means and number of samples for each label
        # label_means = self._get_label_means(labels, self._distributions)
        # label_samples = labels.value_counts()

        # Initialize
        data_index = self._features.index
        noise_indices = []

        # Iterate through labels and see if we should remove values
        for file_id in range(self.n_files):

            # Check length and continue if short
            if sum(data_index.get_level_values(0) == file_id) < self.min_length:
                continue

            # Check distribution and find cut-off
            dist = self._distributions[file_id]
            ind_remove_label = [(file_id, j) for j in np.where(dist < dist.mean())[0]]
            # Extend list to keep track
            noise_indices.extend(ind_remove_label)

            # Verbose
            if len(noise_indices) > 0 and self.verbose > 1:
                filename = (
                    self._metadata[file_id]["file"]
                    if isinstance(self._metadata, dict)
                    else None
                )
                logger.info(
                    f"Removing {len(ind_remove_label)} samples from "
                    f"`{filename or file_id}`"
                )

        # Set noise indices
        self._noise_indices = noise_indices

    # @staticmethod
    # def _get_label_means(labels, dists):
    #     # Init
    #     label_means = {k: [] for k in labels.unique()}
    #
    #     # Append distributions
    #     for i, d in enumerate(dists):
    #         # Skip failed reads
    #         if i not in labels:
    #             continue
    #
    #         # Extend distribution
    #         label_means[labels[(i, 0)]].extend(d)
    #
    #     # Return means
    #     return {k: np.mean(v) for k, v in label_means.items()}

    # --- Deprecation Properties ---

    @property
    def samples(self):
        warnings.warn(
            DeprecationWarning(
                "This pseudo-attribute will be removed in a future version. "
                "Consider using `n_samples` instead."
            )
        )
        return self.n_samples

    @property
    def n_keys(self):
        warnings.warn(
            DeprecationWarning(
                "This pseudo-attribute will be removed in a future version. "
                "Consider using `n_columns` instead."
            )
        )
        return self.n_columns
