#  Copyright (c) 2022 by Amplo.

"""
Feature processor for extracting static features.
"""
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from amplo.automl.feature_processing._base import (
    PERFECT_SCORE,
    BaseFeatureExtractor,
    check_data,
)
from amplo.base.exceptions import NotFittedError

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

    def __init__(
        self, target: str = "", mode: str = "classification", verbose: int = 1
    ):
        super().__init__(target, mode, verbose)

    def fit(self, data: pd.DataFrame, **fit_params):
        # We implement fit_transform because we anyhow transform the data. Therefore,
        # when using fit_transform we don't have to do redundant transformations.
        self.fit_transform(data)
        return self

    def fit_transform(self, data: pd.DataFrame, **fit_params) -> pd.DataFrame:
        self.logger.info("Fitting static feature extractor.")
        check_data(data)

        # Initialize fitting
        x, y = data.drop(self.target, axis=1), data[self.target]
        self.initialize_baseline(x, y)
        assert self._baseline_score is not None
        if self._baseline_score > PERFECT_SCORE:
            self.logger.info("Features are good, we're skipping feature aggregation.")
            self.is_fitted_ = True
            self.skipped_ = True
            return data

        # Fit features
        x_out = pd.concat(
            [
                self._fit_transform_raw_features(x),
                self._fit_transform_cross_features(x, y),
                self._fit_transform_trigo_features(x, y),
                self._fit_transform_inverse_features(x, y),
            ],
            axis=1,
        )

        self.is_fitted_ = True
        return pd.concat([x_out[self.features_], y], axis=1)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Transforming data.")
        check_data(data)
        if not self.is_fitted_:
            raise NotFittedError
        if self.skipped_:
            return data

        # Apply transformations
        x_out = pd.concat(
            [
                self._transform_raw_features(data),
                self._transform_cross_features(data),
                self._transform_trigo_features(data),
                self._transform_inverse_features(data),
            ],
            axis=1,
        )

        # Ensure ordering of columns & sanitize
        if self.target in data:
            return pd.concat([x_out[self.features_], data[self.target]], axis=1)
        return x_out[self.features_]

    # ----------------------------------------------------------------------
    # Feature processing

    @property
    def raw_features_(self) -> list[str]:
        out = [str(c) for c in self.features_ if not re.search(".+__.+", c)]
        return sorted(out)

    def _fit_transform_raw_features(self, x: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Adding {x.shape[1]} raw features.")

        # Add accepted features
        self.add_features(x)

        return x

    def _transform_raw_features(self, x: pd.DataFrame) -> pd.DataFrame:
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
    def cross_features_(self) -> list[str]:
        out = [str(c) for c in self.features_ if re.search("__(mul|div|x|d)__", c)]
        return sorted(out)

    def _fit_transform_cross_features(
        self, x: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
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
                div_score = self.calc_feature_score(div_feature, y)
                if self.accept_feature(div_score):
                    col_a_useless_so_far = False
                    name = f"{col_a}__div__{col_b}"
                    scores[name] = div_score
                    x_out += [div_feature.rename(name)]

                # Make __mul__ feature
                mul_feature = x[col_a] * x[col_b]
                mul_score = self.calc_feature_score(mul_feature, y)
                if self.accept_feature(mul_score):
                    name = "{}__mul__{}".format(*sorted([col_a, col_b]))
                    col_a_useless_so_far = False
                    scores[name] = mul_score
                    x_out += [mul_feature.rename(name)]

        # Decide which features to accept
        selected_scores = self.select_scores(pd.Series(scores))
        x_out_df = (
            pd.concat(x_out, axis=1)[selected_scores.index]
            if x_out
            else pd.DataFrame(index=x.index)
        )
        self.logger.info(f"Accepted {x_out_df.shape[1]} cross features.")

        # Add accepted features
        self.add_features(x_out_df)

        return x_out_df

    def _transform_cross_features(self, x: pd.DataFrame) -> pd.DataFrame:
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
    def trigo_features_(self) -> list[str]:
        out = [str(c) for c in self.features_ if re.match("(sin|cos)__", c)]
        return sorted(out)

    def _fit_transform_trigo_features(self, x: pd.DataFrame, y: pd.Series):
        self.logger.info("Fitting trigonometric features.")

        # Make features
        sin_x = np.sin(x).rename(columns={col: f"sin__{col}" for col in x})
        cos_x = np.cos(x).rename(columns={col: f"cos__{col}" for col in x})
        feats = pd.concat([sin_x, cos_x], axis=1)

        # Score and decide which features to accept
        scores = self.select_scores(
            feats.apply(self.calc_feature_score, y=y, axis=0),
        )
        x_out = feats[scores.index]
        self.logger.info(f"Accepted {x_out.shape[1]} raw features.")

        # Add accepted features
        self.add_features(x_out)

        return x_out

    def _transform_trigo_features(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self.trigo_features_:
            self.logger.debug("No trigonometric features added.")
            return pd.DataFrame(index=x.index)

        self.logger.info("Transforming trigonometric features.")

        # Group by transformation
        feat_info_list = [list(f.partition("__"))[::2] for f in self.trigo_features_]
        feat_info = pd.DataFrame(feat_info_list).groupby(0).agg(list)[1]

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
    def inverse_features_(self) -> list[str]:
        out = [str(c) for c in self.features_ if re.match("inv__", c)]
        return sorted(out)

    def _fit_transform_inverse_features(self, x: pd.DataFrame, y: pd.Series):
        self.logger.info("Fitting inverse features.")

        # Make features
        with np.errstate(divide="ignore"):  # ignore true_divide warnings
            feats = (1.0 / x).rename(columns={col: f"inv__{col}" for col in x})
        feats.fillna(0, inplace=True)

        # Score and decide which features to accept
        scores = self.select_scores(feats.apply(self.calc_feature_score, y=y, axis=0))
        x_out = feats[scores.index]
        self.logger.info(f"Accepted {x_out.shape[1]} inverse features.")

        # Add accepted features
        self.add_features(x_out)

        return x_out

    def _transform_inverse_features(self, x: pd.DataFrame) -> pd.DataFrame:
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
