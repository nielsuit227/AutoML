#  Copyright (c) 2022 by Amplo.
from __future__ import annotations

import faiss
import numpy as np
import pandas as pd

from amplo.base import LoggingMixin


class IntervalAnalyser(LoggingMixin):

    noise = -1

    def __init__(
        self,
        target: str,
        norm: str = "euclidean",  # TODO: implement functionality
        min_length: int = 1000,
        n_neighbors: int | None = None,
        n_trees: int = 10,
        verbose: int = 1,
    ):
        """
        Interval Analyser for Log file classification. Has two purposes:
        - Remove healthy data in longer, faulty logs
        - Remove redundant data in large datasets

        Uses Facebook's FAISS for K-Nearest Neighbors approximation.

        NOTE: This should no longer be used as a standalone module, and should be used
        with the Pipeline. It can still be used as as a standalone module, but it needs
        to take normalized, multi-indexed data.

        Parameters
        ----------
        target : str
            Target column name, must be a folder in the parent folder!
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
        super().__init__(verbose=verbose)
        # Test
        self.available_norms = ["euclidean", "manhattan", "angular", "hamming", "dot"]
        if norm not in self.available_norms:
            raise ValueError(f"Unknown norm, pick from {self.available_norms}")

        # Parameters
        self.target = target
        self.min_length = min_length
        self.norm = norm
        self.n_trees = n_trees
        self.n_neighbors = n_neighbors
        self.verbose = verbose

        # Initializers
        self._engine = None
        self.distributions_ = None
        self.noise_indices_ = None
        self.n_samples_ = None
        self.n_columns_ = None
        self.n_files_ = None
        self.n_folders_ = None

        # Flags
        self.is_fitted_ = False

    def fit_transform(self, data: pd.DataFrame):
        """
        Function that runs the K-Nearest Neighbors and returns a dataframe with the
        sensitivities.

        Parameters
        ----------
        data : pd.DataFrame
            Multi-indexed dataset containing feature (and target) columns
            or path to folder that is correctly structured (see Notes)
        labels : pd.Series, optional
            Multi-indexed dataset containing target columns.

        Returns
        -------
        pd.DataFrame
            Data with noise filtered
        """
        # Validate data
        data = self._validate_data(data)
        self._set_data_attr(data)

        # Set up Annoy Engine (only now that n_keys is known)
        self._build_engine(data)

        # Make distribution
        self.is_fitted_ = True
        return self.transform(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out noise for the provided data frame.
        """
        if not self.is_fitted_:
            raise ValueError("Object not fitted yet.")

        data = self._validate_data(data)
        self._make_distribution(data)
        return self._filter_noise(data)

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the data and sets certain attributes.
        """
        # Validate data
        if not self.target in data:
            raise ValueError(f"Target column {self.target} not in dataframe.")
        if len(data.index.names) != 2:
            raise ValueError(f"Dataframe needs to have a multi-index.")
        if data.max().max() > 10 or data.min().min() < -10:
            raise ValueError("Data needs to be normalized.")

        # Assert that all dtypes are of dtype float32
        #  See: https://github.com/facebookresearch/faiss/issues/461
        for col in data.select_dtypes(exclude=["float32"]).columns:
            data[col] = data[col].astype("float32")

        return data

    def _set_data_attr(self, data: pd.DataFrame):
        """
        Sets data specific class attributes
        """
        self.n_folders_ = data[self.target].nunique()
        self.n_files_ = data.index.get_level_values(0).nunique()
        self.n_samples_, self.n_columns_ = len(data), len(data.keys()) - 1
        if self.n_neighbors is None:
            self.n_neighbors = min(3 * self.n_samples_ // self.n_files_, 5000)

    def _build_engine(self, data: pd.DataFrame):
        """
        Builds the ANNOY engine.

        Parameters
        ----------
        features : pd.DataFrame
            Multi-indexed dataset containing feature columns
        """
        # Create engine
        self.logger.info("Building interval analyser engine.")
        self._engine = faiss.IndexFlatL2(self.n_columns_)

        # Add the data to ANNOY
        data = data.drop(self.target, axis=1)
        self._engine.add(np.ascontiguousarray(data.values))  # type: ignore

    def _make_distribution(self, data: pd.DataFrame):
        """
        Given a build K-Nearest Neighbors, returns the label distribution

        Parameters
        ----------
        features : pd.DataFrame
            Multi-indexed dataset containing feature columns
        labels : pd.Series
            Multi-indexed dataset containing target columns
        """
        assert self._engine and self.n_neighbors
        self.logger.info("Calculating interval within-class distributions.")
        labels = data[self.target]
        data = data.drop(self.target, axis=1)

        # Search nearest neighbors for all samples - has to be iterative for large files
        distribution = []
        for i, row in data.iterrows():
            _, neighbors = self._engine.search(  # type: ignore
                np.ascontiguousarray(np.array(row.values).reshape((1, -1))),
                self.n_neighbors,
            )
            match_mask = labels.iloc[neighbors.reshape(-1)] == labels.loc[i]  # type: ignore
            distribution.append(pd.Series(match_mask).sum() / self.n_neighbors)  # type: ignore

        # Parse into list of lists
        self.distributions_ = pd.Series(distribution, index=labels.index)

    def _filter_noise(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This function selects samples given the calculated distributions. It only
        removes samples from logs which are longer (> min_length), and only the samples
        with lower in-class neighbors.

        One could come up with a fancier logic, using the total dataset samples, the
        class-balance & sample redundancy.
        """
        assert self.distributions_ is not None
        # Verbose
        self.logger.info("Creating filtered dataset")

        # Initialize
        noise_indices = []

        # Iterate through labels and see if we should remove values
        for log in self.distributions_.index.get_level_values(0):

            # Check distribution and find cut-off
            dist = self.distributions_[log]
            ind_remove_label = [(log, j) for j in np.where(dist < dist.mean())[0]]

            # Extend list to keep track
            noise_indices.extend(ind_remove_label)

            # Verbose
            self.logger.info(
                f"Removing {len(ind_remove_label)} samples from " f"`{log}`"
            )

        # Set noise indices
        self.noise_indices_ = noise_indices
        data = data.drop(self.noise_indices_, axis=0)
        return data
