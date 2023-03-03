#  Copyright (c) 2022 by Amplo.

"""
Feature processor for extracting static features.
"""

import re

import polars as pl
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
    target : str | None, optional
        Target column that must be present in data, by default None
    mode : str | None, optional
        Model mode: {"classification", "regression"}, by default None
    verbose : int, optional
        Verbisity for logger, by default 0
    """

    def fit(self, data: pl.DataFrame, index_cols: list[str]):  # type: ignore[override]
        # NOTE: We anyhow have to transform the data. Therefore, when calling
        # 'fit_transform' we do no redundant transformations.
        self.fit_transform(data, index_cols)
        return self

    def fit_transform(self, data: pl.DataFrame, index_cols: list[str]) -> pl.DataFrame:  # type: ignore[override]
        self.logger.info("Fitting static feature extractor.")
        check_data(data)

        # Split data
        x = data.drop([self.target, *index_cols])
        y = data[self.target]

        # Initialize fitting
        self.initialize_baseline(x, y)
        assert self._baseline_score is not None
        if self._baseline_score > PERFECT_SCORE:
            self.logger.info("Features are good, we're skipping feature aggregation.")
            self.is_fitted_ = True
            self.skipped_ = True
            return data

        # Fit-transform
        data = pl.concat(
            [
                # Add index
                data.select(index_cols),
                # Fit and transform features
                self._fit_transform_raw_features(x),
                self._fit_transform_cross_features(x, y),
                self._fit_transform_trigo_features(x, y),
                self._fit_transform_inverse_features(x, y),
                # Add target
                data.select(self.target),
            ],
            how="horizontal",
        )

        self.is_fitted_ = True
        return data

    def transform(self, data: pl.DataFrame, index_cols: list[str]) -> pl.DataFrame:  # type: ignore[override]
        self.logger.info("Transforming data.")

        if not self.is_fitted_:
            raise NotFittedError
        if self.skipped_:
            return data

        # Input checks
        check_data(data)

        # Apply transformations
        data = pl.concat(
            [
                # Add index
                data.select(index_cols),
                # Fit and transform features
                self._transform_raw_features(data),
                self._transform_cross_features(data),
                self._transform_trigo_features(data),
                self._transform_inverse_features(data),
                # Add target iff 'self.target' is present
                data.select(self.target)
                if self.target in data.columns
                else pl.DataFrame(),
            ],
            how="horizontal",
        )

        return data

    # ----------------------------------------------------------------------
    # Feature processing

    @property
    def raw_features_(self) -> list[str]:
        out = [str(c) for c in self.features_ if not re.search(".+__.+", c)]
        return sorted(out)

    def _fit_transform_raw_features(self, x: pl.DataFrame) -> pl.DataFrame:
        self.logger.info(f"Adding {x.shape[1]} raw features.")

        # Accept all features
        self.add_features(x)

        return x[self.raw_features_]

    def _transform_raw_features(self, x: pl.DataFrame) -> pl.DataFrame:
        if not self.raw_features_:
            self.logger.debug("No raw features added.")
            return pl.DataFrame()

        self.logger.info("Transforming raw features.")
        if not set(self.raw_features_).issubset(x.columns):
            raise ValueError("Expected raw features do not match with actual.")

        return x[self.raw_features_]

    @property
    def cross_features_(self) -> list[str]:
        out = [str(c) for c in self.features_ if re.search("__(mul|div|x|d)__", c)]
        return sorted(out)

    def _fit_transform_cross_features(
        self, x: pl.DataFrame, y: pl.Series
    ) -> pl.DataFrame:
        self.logger.info("Fitting cross features.")

        scores = {}
        x_out: list[pl.Series] = []
        for i, col_a in enumerate(tqdm(x.columns)):
            col_a_useless_so_far = True
            for j, col_b in enumerate(x.select(x.columns[i + 1 :]).columns):
                # Skip when same column or `col_a` is potentially useless.
                if col_a == col_b or (
                    j > max(50, x.shape[0] // 3) and col_a_useless_so_far
                ):
                    continue

                # Make __div__ feature
                safe_col_b = x[col_b].apply(lambda x: 1e-10 if x == 0 else x)
                div_feature = x[col_a] / safe_col_b
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
        selected_cols = self.select_scores(scores)
        x_out_df = pl.DataFrame(x_out).select(selected_cols)
        self.logger.info(f"Accepted {x_out_df.shape[1]} cross features.")

        # Add accepted features
        self.add_features(x_out_df)

        return x_out_df[self.cross_features_]

    def _transform_cross_features(self, x: pl.DataFrame) -> pl.DataFrame:
        if not self.cross_features_:
            self.logger.debug("No cross features added.")
            return pl.DataFrame()

        self.logger.info("Transforming cross features.")

        x_out: list[pl.Series] = []
        for feature_name in self.cross_features_:
            # Deprecation support
            if "__x__" in feature_name:
                col_a, col_b = feature_name.split("__x__")
                feat = x[col_a] * x[col_b]
                x_out += [feat.rename(feature_name)]
            elif "__d__" in feature_name:
                col_a, col_b = feature_name.split("__d__")
                safe_col_b = x[col_b].apply(lambda x: 1e-10 if x == 0 else x)
                feat = x[col_a] / safe_col_b
                x_out += [feat.rename(feature_name)]
            # New names
            elif "__mul__" in feature_name:
                col_a, col_b = feature_name.split("__mul__")
                feat = x[col_a] * x[col_b]
                x_out += [feat.rename(feature_name)]
            elif "__div__" in feature_name:
                col_a, col_b = feature_name.split("__div__")
                safe_col_b = x[col_b].apply(lambda x: 1e-10 if x == 0 else x)
                feat = x[col_a] / safe_col_b
                x_out += [feat.rename(feature_name)]
            else:
                raise ValueError(f"Cross feature not recognized: {feature_name}")

        x_out_df = pl.DataFrame(x_out)

        assert set(self.cross_features_) == set(
            x_out_df.columns
        ), "Expected cross features do not match with actual."

        return x_out_df[self.cross_features_]

    @property
    def trigo_features_(self) -> list[str]:
        out = [str(c) for c in self.features_ if re.match("(sin|cos)__", c)]
        return sorted(out)

    def _fit_transform_trigo_features(
        self, x: pl.DataFrame, y: pl.Series
    ) -> pl.DataFrame:
        self.logger.info("Fitting trigonometric features.")

        # Make features
        sin_x = x.select([pl.col(c).sin().alias(f"sin__{c}") for c in x.columns])
        cos_x = x.select([pl.col(c).cos().alias(f"cos__{c}") for c in x.columns])
        feats = pl.concat([sin_x, cos_x], how="horizontal")

        # Score
        scores: dict[str, float] = {}
        for col in feats.columns:
            scores[col] = self.calc_feature_score(feats[col], y)

        # Decide which features to accept
        selected_cols = self.select_scores(scores)
        x_out = feats.select(selected_cols)
        self.logger.info(f"Accepted {x_out.shape[1]} trigo features.")

        # Add accepted features
        self.add_features(x_out)

        return x_out[self.trigo_features_]

    def _transform_trigo_features(self, x: pl.DataFrame) -> pl.DataFrame:
        if not self.trigo_features_:
            self.logger.debug("No trigonometric features added.")
            return pl.DataFrame()

        self.logger.info("Transforming trigonometric features.")

        # Group by transformation
        feat_info_list = [list(f.partition("__"))[::2] for f in self.trigo_features_]
        feat_info = (
            pl.DataFrame(feat_info_list, orient="row")
            .groupby("column_0")
            .agg(pl.col("column_1"))
        )

        # Transform
        x_out: list[pl.Expr] = []
        for func, cols in feat_info.rows():
            # getattr(...) gets the pl.Expr for the function
            x_out.extend(getattr(pl.col(c), func)().alias(f"{func}__{c}") for c in cols)
        x_out_df = x.select(x_out)

        assert set(self.trigo_features_) == set(
            x_out_df.columns
        ), "Expected trigonometric features do not match with actual."

        return x_out_df[self.trigo_features_]

    @property
    def inverse_features_(self) -> list[str]:
        out = [str(c) for c in self.features_ if re.match("inv__", c)]
        return sorted(out)

    def _fit_transform_inverse_features(
        self, x: pl.DataFrame, y: pl.Series
    ) -> pl.DataFrame:
        self.logger.info("Fitting inverse features.")

        # Make features
        feats = x.select([pl.col(c).pow(-1).alias(f"inv__{c}") for c in x.columns])
        feats = feats.with_column(
            pl.when(pl.all().is_infinite()).then(0.0).otherwise(pl.all()).keep_name()
        )

        # Score
        scores: dict[str, float] = {}
        for col in feats.columns:
            scores[col] = self.calc_feature_score(feats[col], y)

        # Decide which features to accept
        selected_cols = self.select_scores(scores)
        x_out = feats.select(selected_cols)
        self.logger.info(f"Accepted {x_out.shape[1]} inverse features.")

        # Add accepted features
        self.add_features(x_out)

        return x_out[self.inverse_features_]

    def _transform_inverse_features(self, x: pl.DataFrame) -> pl.DataFrame:
        if not self.inverse_features_:
            self.logger.debug("No inverse features added.")
            return pl.DataFrame()

        self.logger.info("Transforming inverse features.")

        # Get all columns to invert
        inv_columns = [
            f[len("inv__") :] for f in self.inverse_features_  # remove prefix
        ]

        # Transform
        x_out = x.select([pl.col(c).pow(-1).alias(f"inv__{c}") for c in inv_columns])
        x_out = x_out.with_column(
            pl.when(pl.all().is_infinite()).then(0.0).otherwise(pl.all()).keep_name()
        )

        assert set(self.inverse_features_) == set(
            x_out.columns
        ), "Expected inverse features do not match with actual."

        return x_out[self.inverse_features_]
