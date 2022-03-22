import os
import copy
import faiss
import numpy as np
import pandas as pd
from Amplo.AutoML import DataProcesser
from Amplo.AutoML import FeatureProcesser


class IntervalAnalyser:

    def __init__(self,
                 folder: str = None,
                 norm: str = 'euclidean',
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
        To use this interval analyser, make sure that your logs are located in a folder of their label, with one parent
        folder with all labels, e.g.:
        +-- Parent Folder
        |   +-- Label_1
        |       +-- Log_1.*
        |       +-- Log_2.*
        |   +-- Label_2
        |       +-- Log_3.*

        Parameters
        ----------
        folder [str]:       Parent folder of labels
        index_col [str]:    For reading the log files
        min_length [int]:   Minimum length to cut off, everything shorter is left untouched
        norm [str]:         Optimization metric for K-Nearest Neighbors
        n_neighbors [int]:  Quantity of neighbors, default to 3 * log length
        n_trees [int]:      Quantity of trees
        """
        # Parameters
        self.folder = folder + '/' if len(folder) == 0 or folder[-1] != '/' else folder
        self.min_length = min_length
        self.norm = norm
        self.n_trees = n_trees
        self.n_neighbors = n_neighbors
        self.verbose = verbose

        # Initializers
        self.keys = None
        self.n_keys = None
        self.samples = None
        self.avg_samples = None
        self.n_files = 0
        self.n_folders = 0
        self._mins = pd.DataFrame(index=[0])
        self._maxs = pd.DataFrame(index=[0])
        self._labels = None
        self._engine = None
        self._distributions = None
        self._str_labels = None

        # Test
        assert norm in ['euclidean', 'manhattan', 'angular', 'hamming', 'dot']

    def fit_transform(self) -> pd.DataFrame:
        """
        Function that runs the K-Nearest Neighbors and returns a NumPy array with the sensitivities.

        Returns
        -------
        np.array: Estimation of strength of correlation
        """
        # Get data
        df, labels = self._parse_data()

        # Set up Annoy Engine (only now that n_keys is known)
        self._engine = self._build_engine(df)

        # Make distribution
        self._distributions = self._make_distribution(df, labels)

        # Return new dataset
        return self._create_dataset(df, labels)

    def _create_dataset(self, df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """
        This function selects samples given the calculated distributions. It only removes samples from logs which are
        longer (> min_length), and only the samples with lower in-class neighbors.

        One could come up with a fancier logic, using the total dataset samples, the class-balance & sample redundancy.

        parameters
        ----------
        """
        # Verbose
        if self.verbose > 0:
            print('[AutoML] Creating filtered dataset')

        # Get in-class means and number of samples for each label
        # label_means = self._get_label_means(labels, self._distributions)
        # label_samples = labels.value_counts()

        # Initialize
        ind_remove = []

        # Iterate through labels and see if we should remove values
        for i in range(self.n_files):

            # Check length and continue if short
            if sum(df.index.get_level_values(0) == i) < self.min_length:
                continue

            # Check distribution and find cut-off
            dist = self._distributions[i]
            ind_remove.extend([(i, j) for j in np.where(dist < dist.mean())[0]])

            # Verbose
            if len(ind_remove) > 0 and self.verbose > 1:
                print(f'[AutoML] Removing {len(ind_remove)} samples from {os.listdir(self.folder)[labels[(i, 0)]]}')

        # Remove samples from df
        df = df.drop(ind_remove, axis=0)

        # Return original scales
        df = df * (self._maxs[df.keys()] - self._mins[df.keys()]) + self._mins[df.keys()]

        # Add labels
        df['labels'] = self._str_labels

        return df

    def _parse_data(self):
        """
        Reads all files and sets a multi index

        Returns
        -------
        pd.DataFrame: all log files
        """
        # Verbose
        if self.verbose > 0:
            print('[AutoML] Parsing data for interval analyser.')

        # Result init
        dfs = []

        # Loop through files
        for folder in os.listdir(self.folder):
            for file in os.listdir(self.folder + folder):

                # Verbose
                if self.verbose > 1:
                    print(f"[AutoML] {self.folder}{folder}/{file}")

                # Read df
                df = self._read(f'{self.folder}{folder}/{file}')

                # Set label
                df['labels'] = folder

                # Set second index
                df = df.set_index(pd.MultiIndex.from_product([[self.n_files], df.index.values], names=['log', 'index']))

                # Add to list
                dfs.append(df)

                # Increment
                self.n_files += 1
            self.n_folders += 1

        # Concatenate dataframes
        dfs = pd.concat(dfs)

        # Extract string labels
        self._str_labels = copy.deepcopy(dfs['labels'])

        # Clean data
        dp = DataProcesser(missing_values='zero', target='labels')
        dfs = dp.fit_transform(dfs)

        # Remove datetime columns
        if len(dp.date_cols) != 0:
            dfs = dfs.drop(dp.date_cols, axis=1)

        # Remove classes
        labels = dfs['labels']
        dfs = dfs.drop('labels', axis=1)

        # Select features -- bit hacky to avoid memory
        fp = FeatureProcesser(extract_features=False, mode='classification')
        fp.x, fp.y = dfs, labels
        features, _ = fp._sel_gini_impurity()
        dfs = dfs[features]

        # Normalize
        self._mins, self._maxs = dfs.min(), dfs.max()
        dfs = (dfs - dfs.min()) / (dfs.max() - dfs.min())

        # Set sizes
        self.samples = len(dfs)
        self.n_keys = len(dfs.keys())
        if self.n_neighbors is None:
            self.n_neighbors = min(int(3 * self.samples / self.n_files), 5000)

        # Return
        return dfs, labels

    def _build_engine(self, df):
        """
        Builds the ANNOY engine.
        """
        # Create engine
        if self.verbose > 0:
            print('[AutoML] Building interval analyser engine.')
        engine = faiss.IndexFlatL2(self.n_keys)

        # Add the data to ANNOY
        engine.add(np.ascontiguousarray(df.values))

        return engine

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

    def _make_distribution(self, df: pd.DataFrame, labels: pd.Series) -> pd.Series:
        """
        Given a build K-Nearest Neighbors, returns the label distribution
        """
        if self.verbose > 0:
            print('[AutoML] Calculating interval within-class distributions.')

        # Search nearest neighbors for all samples -- has to be iterative for large files -.-
        distribution = []
        for i, row in df.iterrows():
            _, neighbors = self._engine.search(np.ascontiguousarray(row.values.reshape((1, -1))), self.n_neighbors)
            distribution.append(sum(labels.iloc[neighbors.reshape(-1)] == labels.loc[i]) / self.n_neighbors)

        # Parse into list of lists
        return pd.Series(distribution, index=labels.index)

    @staticmethod
    def _get_label_means(labels, dists):
        # Init
        label_means = {k: [] for k in labels.unique()}

        # Append distributions
        for i, d in enumerate(dists):
            # Skip failed reads
            if i not in labels:
                continue

            # Extend distribution
            label_means[labels[(i, 0)]].extend(d)

        # Return means
        return {k: np.mean(v) for k, v in label_means.items()}
