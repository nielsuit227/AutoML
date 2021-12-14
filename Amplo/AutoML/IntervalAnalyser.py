import os
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from Amplo.AutoML import DataProcesser


class IntervalAnalyser:

    def __init__(self,
                 folder: str = None,
                 norm: str = 'euclidean',
                 n_neighbors: int = None,
                 n_trees: int = 10,
                 ):
        """
        Interval Analyser for Log file classification. When log files have to be classifier, and there is not enough
        data for time series methods (such as LSTMs, ROCKET or Weasel, Boss, etc), one needs to fall back to classical
        machine learning models which work better with lower samples. But this raises the problem of which samples to
        classify. You shouldn't just simply classify on every sample and accumulate, that may greatly disrupt
        classification performance. Therefore, we introduce this interval analyser. By using K-Nearest Neighbors,
        one can estimate the strength of correlation for every sample inside a log. Using this allows for better
        interval selection for classical machine learning models.

        To use this interval analyser, make sure:
        - That your logs are of equal length
        - That your logs have identical keys
        - That your logs are located in a folder of their class, with one parent folder with all classes, e.g.:
        +-- Parent Folder
        |   +-- Class_1
        |       +-- Log_1.*
        |       +-- Log_2.*
        |   +-- Class_2
        |       +-- Log_3.*

        Parameters
        ----------
        folder [str]:         Parent folder of classes
        index_col [str]:      For reading the log files
        norm [str]:           Optimization metric for K-Nearest Neighbors
        n_neighbors [int]:    Quantity of neighbors, default to 3 * log length
        n_trees [int]:        Quantity of trees
        """
        # Parameters
        self.folder = folder + '/' if len(folder) == 0 or folder[-1] != '/' else folder
        self.norm = norm
        self.n_trees = n_trees
        self.n_samples, self.n_keys = self._check_length()
        self.n_neighbors = 3 * self.n_samples if n_neighbors is None else n_neighbors

        # Initializers
        self.n_files = 0
        self.n_folders = 0
        self.means = None
        self.stds = None

        # Test
        assert norm in ['euclidean', 'manhattan', 'angular', 'hamming', 'dot']

    def analyse(self) -> np.ndarray:
        """
        Function that runs the K-Nearest Neighbors and returns a NumPy array with the sensitivities.

        Returns
        -------
        np.array: Estimation of strength of correlation
        """
        # Set up
        engine = AnnoyIndex(self.n_keys, self.norm)

        # Get data
        df, labels = self._parse_data()

        # Index
        for i, row in df.iterrows():
            engine.add_item(i[0] * self.n_samples + i[1], row.values)

        # Build
        engine.build(self.n_trees)

        # Return distribution
        return self._make_distribution(engine, df, labels)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def _check_length(self) -> [int, int]:
        """
        Detects the extension from the log files
        """
        # Read first log
        folder = os.listdir(self.folder)[0]
        file = os.listdir(self.folder + folder)[0]
        df = self._read(f"{self.folder}{folder}/{file}")

        # Set lengths
        length, keys = len(df), len(df.keys())

        return length, keys

    def _read(self, path: str) -> pd.DataFrame:
        """
        Wrapper for various read functions
        """
        f_ext = path[path.rfind('.'):]
        if f_ext == '.csv':
            return pd.read_csv(path)
        elif f_ext == '.json':
            return pd.read_json(path)
        elif f_ext == '.xml':
            return pd.read_xml(path)
        elif f_ext == '.feather':
            return pd.read_feather(path)
        elif f_ext == '.parquet':
            return pd.read_parquet(path)
        elif f_ext == '.stata':
            return pd.read_stata(path)
        elif f_ext == '.pickle':
            return pd.read_pickle(path)
        else:
            raise NotImplementedError('File format not supported.')

    def _parse_data(self) -> [pd.DataFrame, pd.Series]:
        """
        Reads all files and sets a multi index
        Returns
        -------
        pd.DataFrame: all log files
        """
        # Result init
        dfs = []

        # Loop through files
        for folder in os.listdir(self.folder):
            for file in os.listdir(self.folder + folder):

                # Read df
                df = self._read(f'{self.folder}{folder}/{file}')

                # Set label
                df['class'] = folder

                # Set second index
                df = df.set_index(pd.MultiIndex.from_product([[self.n_files], df.index.values], names=['log', 'index']))

                # Add to list
                dfs.append(df)

                # Increment
                self.n_files += 1
            self.n_folders += 1

        # Concatenate dataframes
        dfs = pd.concat(dfs)

        # Remove classes
        classes = dfs['class']
        dfs = dfs.drop('class', axis=1)
        dp = DataProcesser(missing_values='zero')
        dfs = dp.fit_transform(dfs)

        # Remove datetime columns
        if len(dp.date_cols) != 0:
            dfs = dfs.drop(dp.date_cols, axis=1)

        # Return
        return dfs, classes

    def _make_distribution(self, engine, df: pd.DataFrame, classes: pd.Series) -> np.ndarray:
        """
        Given a build K-Nearest Neighbors, returns the label distribution
        """
        # Result setup
        distribution = [[] for _ in range(self.n_files)]

        # Iterate through samples
        for (log, sample), data in df.iterrows():
            # Get neighbor indices
            neighbor_inds = engine.get_nns_by_vector(data.values, self.n_neighbors)

            # Set percentage
            distribution[log].append((classes.iloc[neighbor_inds] == classes[(log, sample)]).sum() / self.n_neighbors)

        # Mask
        lens = np.array([len(item) for item in distribution])
        mask = lens[:, None] > np.arange(lens.max())
        out = np.full(mask.shape, 0)
        out[mask] = np.concatenate(distribution)
        distribution = np.array(out)

        # Average over logs
        return np.mean(distribution * 100, axis=0)
