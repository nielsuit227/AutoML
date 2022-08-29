#  Copyright (c) 2022 by Amplo.

"""
Feature processor for extracting static features.
"""

import json
import re
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from amplo.automl.feature_processing._base import (
    BaseFeatureExtractor,
    sanitize_dataframe,
)

__all__ = ["StaticFeatureExtractor"]


class StaticFeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor for static data.

    Parameters
    ----------
    mode : str
        Model mode: {"classification", "regression"}.
    verbose : int
        Verbosity for logger.
    """

    _add_to_settings = [
        "means_",
        "stds_",
        "centers_",
        *BaseFeatureExtractor._add_to_settings,
    ]

    def __init__(self, mode="notset", verbose=0):
        super().__init__(mode=mode, verbose=verbose)

    def _fit_transform(self, x, y=None, **fit_params):
        self.logger.info("Start fitting data.")

        # Input checks
        x, y = self._check_x_y(x, y)
        numeric_cols = [
            col for col, typ in zip(x, x.dtypes) if np.issubdtype(typ, np.number)
        ]
        if set(numeric_cols) != set(x):
            warn(
                "Handling non-numeric data is (currently) not supported. "
                "Corresponding columns will be ignored.",
                UserWarning,
            )
            x = x[numeric_cols]

        # Initialize fitting
        self._set_validation_model()
        self._init_feature_baseline_scores(x, y)

        # Fit features
        x_out = pd.concat(
            [
                self._fit_transform_raw_features(x),
                self._fit_transform_cross_features(x, y, update_baseline=True),
                self._fit_transform_k_means_features(x, y, update_baseline=True),
                self._fit_transform_trigo_features(x, y, update_baseline=True),
                self._fit_transform_inverse_features(x, y, update_baseline=True),
            ],
            axis=1,
        )

        self.logger.info("Finished fitting.")
        return sanitize_dataframe(x_out[self.features_])

    def _transform(self, x, y=None):
        self.logger.info("Transforming data.")

        # Handle input
        x = self._check_x(x)

        # Apply transformations
        x_out = pd.concat(
            [
                self._transform_raw_features(x),
                self._transform_cross_features(x),
                self._transform_k_means_features(x),
                self._transform_trigo_features(x),
                self._transform_inverse_features(x),
            ],
            axis=1,
        )

        return sanitize_dataframe(x_out[self.features_])

    # ----------------------------------------------------------------------
    # Feature processing

    @property
    def raw_features_(self):
        out = [str(c) for c in self.features_ if not re.search(".+__.+", c)]
        return sorted(out)

    def _fit_transform_raw_features(self, x, y=None):
        self.logger.info(f"Adding {x.shape[1]} raw features.")

        # Add accepted features
        self.add_features(x)

        return x

    def _transform_raw_features(self, x):
        if not self.raw_features_:
            self.logger.debug("No raw features added.")
            return pd.DataFrame(index=x.index)

        self.logger.info("Transforming raw features.")

        assert set(self.raw_features_).issubset(
            x
        ), "Expected raw features do not match with actual."
        x_out = x[self.raw_features_]

        return x_out

    @property
    def cross_features_(self):
        out = [str(c) for c in self.features_ if re.search("__(mul|div|x|d)__", c)]
        return sorted(out)

    def _fit_transform_cross_features(self, x, y, update_baseline=True):
        self.logger.info("Fitting cross features.")

        scores = {}
        x_out = []  # empty df for concat (later)
        for i, col_a in enumerate(tqdm(x)):
            col_a_useless_so_far = True
            for j, col_b in enumerate(x.iloc[:, i + 1 :]):
                # Skip when same column or `col_a` is potentially useless.
                if col_a == col_b or (
                    j > max(50, x.shape[0] // 3) and col_a_useless_so_far
                ):
                    continue

                # Make __div__ feature
                div_feature = x[col_a] / x[col_b].replace(0, 1e-10)
                div_score = self._calc_feature_scores(div_feature, y)
                if self.accept_feature(div_score):
                    col_a_useless_so_far = False
                    name = f"{col_a}__div__{col_b}"
                    scores[name] = div_score
                    x_out += [div_feature.rename(name)]

                # Make __mul__ feature
                mul_feature = x[col_a] * x[col_b]
                mul_score = self._calc_feature_scores(mul_feature, y)
                if self.accept_feature(mul_score):
                    name = "{}__mul__{}".format(*sorted([col_a, col_b]))
                    col_a_useless_so_far = False
                    scores[name] = mul_score
                    x_out += [mul_feature.rename(name)]

        # Decide which features to accept
        scores = self.select_scores(
            pd.DataFrame(scores), best_n_per_class=50, update_baseline=update_baseline
        )
        x_out = (
            pd.concat(x_out, axis=1)[scores.columns]
            if x_out
            else pd.DataFrame(index=x.index)
        )
        self.logger.info(f"Accepted {x_out.shape[1]} cross features.")

        # Add accepted features
        self.add_features(x_out)

        return x_out

    def _transform_cross_features(self, x):
        if not self.cross_features_:
            self.logger.debug("No cross features added.")
            return pd.DataFrame(index=x.index)

        self.logger.info("Transforming cross features.")

        x_out = []
        for feature_name in self.cross_features_:
            # Deprecation support
            if "__x__" in feature_name:
                col_a, col_b = feature_name.split("__x__")
                feat = x[col_a] * x[col_b]
                x_out += [feat.rename(feature_name)]
            elif "__d__" in feature_name:
                col_a, col_b = feature_name.split("__d__")
                feat = x[col_a] / x[col_b].replace(0, -1e-10)
                x_out += [feat.rename(feature_name)]
            # New names
            elif "__mul__" in feature_name:
                col_a, col_b = feature_name.split("__mul__")
                feat = x[col_a] * x[col_b]
                x_out += [feat.rename(feature_name)]
            elif "__div__" in feature_name:
                col_a, col_b = feature_name.split("__div__")
                feat = x[col_a] / x[col_b].replace(0, -1e-10)
                x_out += [feat.rename(feature_name)]
            else:
                raise ValueError(f"Cross feature not recognized: {feature_name}")

        x_out = pd.concat(x_out, axis=1)

        assert set(self.cross_features_) == set(
            x_out
        ), "Expected cross features do not match with actual."

        return x_out

    @property
    def k_means_features_(self):
        out = [str(c) for c in self.features_ if re.match("dist__", c)]
        return sorted(out)

    def _fit_transform_k_means_features(self, x, y, update_baseline=True):
        self.logger.info("Fitting k-Means features.")

        # Prepare data
        means, stds = x.mean(), x.std().replace(0, 1)
        self.means_ = json.loads(means.to_json())
        self.stds_ = json.loads(stds.to_json())
        x = (x - means) / stds

        # Determine clusters
        n_clusters = min(x.shape[1], max(8, int(np.log10(x.shape[1] * 8))))
        k_means = MiniBatchKMeans(n_clusters=n_clusters)
        col_names = [f"dist__{c}_{n_clusters}" for c in range(n_clusters)]
        distances = pd.DataFrame(
            k_means.fit_transform(x), columns=col_names, index=x.index
        )
        distances = sanitize_dataframe(distances)
        centers = pd.DataFrame(k_means.cluster_centers_, columns=x.columns)
        self.centers_ = json.loads(centers.to_json())

        # Score and decide which features to accept
        scores = self.select_scores(
            distances.apply(self._calc_feature_scores, y=y),
            best_n_per_class=50,
            update_baseline=update_baseline,
        )
        x_out = distances[scores.columns]
        self.logger.info(f"Accepted {x_out.shape[1]} k-Means features.")

        # Add accepted features
        self.add_features(x_out)

        return x_out

    def _transform_k_means_features(self, x):
        if not self.k_means_features_:
            self.logger.debug("No k-Means features added.")
            return pd.DataFrame(index=x.index)

        self.logger.info("Transforming k-Means features.")

        # Init
        x = (x - pd.Series(self.means_)) / pd.Series(self.stds_)  # normalize
        centers = pd.DataFrame(self.centers_)

        # Extract features
        x_out = []
        for feature_name in self.k_means_features_:
            cluster_idx, _ = feature_name[len("dist__") :].split("_")  # remove prefix
            cluster_center = centers.iloc[int(cluster_idx)]
            feat = ((x - cluster_center) ** 2).sum(1) ** 0.5
            x_out += [feat.rename(feature_name)]
        x_out = pd.concat(x_out, axis=1)

        assert set(self.k_means_features_) == set(
            x_out
        ), "Expected k-Means features do not match with actual."

        return x_out

    @property
    def trigo_features_(self):
        out = [str(c) for c in self.features_ if re.match("(sin|cos)__", c)]
        return sorted(out)

    def _fit_transform_trigo_features(self, x, y, update_baseline=True):
        self.logger.info("Fitting trigonometric features.")

        # Make features
        sin_x = np.sin(x).rename(columns={col: f"sin__{col}" for col in x})
        cos_x = np.cos(x).rename(columns={col: f"cos__{col}" for col in x})
        feats = pd.concat([sin_x, cos_x], axis=1)

        # Score and decide which features to accept
        scores = self.select_scores(
            feats.apply(self._calc_feature_scores, y=y, axis=0),
            best_n_per_class=50,
            update_baseline=update_baseline,
        )
        x_out = feats[scores.columns]
        self.logger.info(f"Accepted {x_out.shape[1]} raw features.")

        # Add accepted features
        self.add_features(x_out)

        return x_out

    def _transform_trigo_features(self, x):
        if not self.trigo_features_:
            self.logger.debug("No trigonometric features added.")
            return pd.DataFrame(index=x.index)

        self.logger.info("Transforming trigonometric features.")

        # Group by transformation
        feat_info = [list(f.partition("__"))[::2] for f in self.trigo_features_]
        feat_info = pd.DataFrame(feat_info).groupby(0).agg(list)[1]

        # Transform
        x_out = []
        for func, cols in feat_info.iteritems():
            col_names = {col: f"{func}__{col}" for col in x}
            x_out += [getattr(np, func)(x[cols]).rename(columns=col_names)]
        x_out = pd.concat(x_out, axis=1)

        assert set(self.trigo_features_) == set(
            x_out
        ), "Expected trigonometric features do not match with actual."

        return x_out

    @property
    def inverse_features_(self):
        out = [str(c) for c in self.features_ if re.match("inv__", c)]
        return sorted(out)

    def _fit_transform_inverse_features(self, x, y, update_baseline=True):
        self.logger.info("Fitting inverse features.")

        # Make features
        with np.errstate(divide="ignore"):  # ignore true_divide warnings
            feats = (1.0 / x).rename(columns={col: f"inv__{col}" for col in x})
        feats = sanitize_dataframe(feats)  # remove np.inf for scoring

        # Score and decide which features to accept
        scores = self.select_scores(
            feats.apply(self._calc_feature_scores, y=y, axis=0),
            best_n_per_class=50,
            update_baseline=update_baseline,
        )
        x_out = feats[scores.columns]
        self.logger.info(f"Accepted {x_out.shape[1]} inverse features.")

        # Add accepted features
        self.add_features(x_out)

        return x_out

    def _transform_inverse_features(self, x):
        if not self.inverse_features_:
            self.logger.debug("No inverse features added.")
            return pd.DataFrame(index=x.index)

        self.logger.info("Transforming inverse features.")

        # Get all columns to invert
        inv_columns = [
            f[len("inv__") :] for f in self.inverse_features_  # remove prefix
        ]

        # Transform
        with np.errstate(divide="ignore"):  # ignore true_divide warnings
            x_out = (1.0 / x[inv_columns]).rename(
                columns={col: f"inv__{col}" for col in x}
            )

        assert set(self.inverse_features_) == set(
            x_out
        ), "Expected inverse features do not match with actual."

        return x_out
