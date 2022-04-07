import scipy.stats
import numpy as np
import pandas as pd


__all__ = ['DummyDataSampler', 'make_data', 'make_cat_data', 'make_num_data']


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
    pd.DataFrame, dict of {str : list}
    """

    # Handle input
    def parse_input(input_, default_value, list_of_list=False, list_of_str=False):
        if input_ is None or input_ is True:
            return default_value
        elif not input_:
            return []  # leave it empty
        elif (list_of_list and not isinstance(input_[0], list)) or \
                (list_of_str and not isinstance(input_, list)):
            return [input_]
        return input_

    cat_choices = parse_input(cat_choices, [list('abc'), list('xyz')], list_of_list=True)
    num_dists = parse_input(num_dists, ['uniform::0::100', 'norm', 'expon::0.4', 'gamma::0.3', 'beta::0.2::0.4'],
                            list_of_str=True)
    assert any([cat_choices, num_dists]), 'Please specify at least one categorical or numerical column'

    # Sample categorical data
    cat_df = pd.DataFrame()
    for i, choice in enumerate(cat_choices):
        cat_df[f'cat_{i}'] = np.random.choice(choice, (num_samples,))

    # Sample numerical data
    num_df = pd.DataFrame()
    for i, dist in enumerate(num_dists):
        # Convert string to function
        if isinstance(dist, str):
            splits = dist.split('::')
            dist = splits[0]
            args = [float(arg) for arg in splits[1:]]
            dist = getattr(scipy.stats, dist)(*args)
        # Add distribution to corresponding array
        num_df[f'num_{i}'] = DummyDataSampler(dist).sample_data(num_samples)

    # Concatenate all data (which is not empty)
    df = pd.concat([df_ for df_ in (cat_df, num_df) if not df_.empty], axis=1)

    return df, dict(cat_cols=cat_df.columns, num_cols=num_df.columns)


def make_cat_data(num_samples: int, cat_choices=None):
    return make_data(num_samples, cat_choices=cat_choices, num_dists=False)


def make_num_data(num_samples: int, num_dists=None):
    return make_data(num_samples, cat_choices=False, num_dists=num_dists)
