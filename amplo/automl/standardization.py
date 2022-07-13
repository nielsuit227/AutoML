#  Copyright (c) 2022 by Amplo.

import pandas as pd


class Standardizer:
    def __init__(self, float_cols: list = None):
        self.float_cols = float_cols
        self.means = None
        self.stds = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fit Input
        self.fit(df)
        df = self.transform(df)
        return df

    def fit(self, df: pd.DataFrame):
        self.means = df[self.float_cols].mean(axis=0)
        self.stds = df[self.float_cols].std(axis=0)
        self.stds[abs(self.stds) <= 1e-9] = 1

    def transform(self, df: pd.DataFrame):
        df[self.float_cols] = (df[self.float_cols] - self.means) / self.stds
        return df
