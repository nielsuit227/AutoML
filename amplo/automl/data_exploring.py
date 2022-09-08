#  Copyright (c) 2022 by Amplo.

import os
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm

from amplo.utils.logging import logger


class DataExplorer:
    def __init__(
        self,
        data,
        y=None,
        mode: str = "regression",
        plot_time_plots: bool = True,
        plot_box_plots: bool = False,
        plot_missing_values: bool = True,
        plot_seasonality: bool = False,
        plot_co_linearity: bool = True,
        plot_differencing: bool = False,
        plot_signal_correlations: bool = False,
        plot_feature_importance: bool = True,
        plot_scatter_plots: bool = False,
        differ: int = 0,
        pre_tag: str = "",
        max_samples: int = 10000,
        season_periods: int = 24,
        lags: int = 60,
        skip_completed: bool = True,
        folder: str = "",
        version: int = 0,
    ):
        """
        Automated Exploratory Data Analysis.
        Covers binary classification and regression.
        It generates:
        - Missing Values Plot
        - Line Plots of all features
        - Box plots of all features
        - Co-linearity Plot
        - SHAP Values
        - Random Forest Feature Importance
        - Predictive Power Score

        Additionally fFor Regression:
        - Seasonality Plots
        - Differentiated Variance Plot
        - Auto Correlation Function Plot
        - Partial Auto Correlation Function Plot
        - Cross Correlation Function Plot
        - Scatter Plots

        All figures are organized per project, category and version with the following
        file structure:
        |-- "Folder parameter"
        |   |-- EDA
        |   |   |-- Timeplots
        |   |   |   |-- v"i"
        |   |   |   |   |-- Feature_0.png
        |   |   |   |   |-- Feature_1.png
        ...

        Parameters
        ----------
        data : pd.DataFrame
            Input data, either Numpy or Pandas
        y : pd.Series
            Output / Target, either Numpy or Pandas
        mode : str
            classification or regression
        plot_time_plots : bool
            Boolean to create time plots or not
        plot_box_plots : bool
            Boolean to create box plots
        plot_missing_values : bool
            Boolean to create missing values plot
        plot_seasonality : bool
            Boolean to create seasonality plots
        plot_co_linearity : bool
            Boolean to create co-linearity plots
        plot_differencing : bool
            Boolean to create differencing plots
        plot_signal_correlations : bool
            Boolean to create signal correlation plots (ACF, PACF, CCF)
        plot_feature_importance : bool
            Boolean to create feature importance plots (SHAP, RF, PPS)
        plot_scatter_plots : bool
            Boolean to create scatter plots
        differ : int
            Differencing order for signal correlation plots
        pre_tag : str
            String to format file names
        max_samples : int
            Optional, under-sampling for large datasets
        season_periods : int
            Optional, season periods for seasonality plots
        lags : int
            Optional, maximum lags to take into account for signal plots
        skip_completed : bool
            Whether to create existing plots
        folder : str
            Main folder to dump files in
        version : int
            All graphs are put in a versioned folder
        """
        assert isinstance(data, np.ndarray) or isinstance(
            data, pd.DataFrame
        ), "Data should be Numpy or Pandas"
        if y is not None:
            assert isinstance(y, np.ndarray) or isinstance(
                y, pd.Series
            ), "Y should be Numpy or Pandas"
        if type(data) != pd.DataFrame:
            data = pd.DataFrame(
                data=data,
                columns=["Feature_{}".format(i) for i in range(data.shape[1])],
            )
        assert mode in ["classification", "regression"]

        # Running booleans
        self.plotTimePlots = plot_time_plots
        self.plotBoxPlots = plot_box_plots
        self.plotMissingValues = plot_missing_values
        self.plotSeasonality = plot_seasonality
        self.plotCoLinearity = plot_co_linearity
        self.plotDifferencing = plot_differencing
        self.plotSignalCorrelations = plot_signal_correlations
        self.plotFeatureImportance = plot_feature_importance
        self.plotScatterPlots = plot_scatter_plots

        # Register data
        self.data = data.astype("float64").fillna(0)
        if y is not None:
            if type(y) == pd.DataFrame:
                y = y[y.keys()[0]]
            if type(y) != pd.Series:
                y = pd.Series(y)
            self.Y = y.astype("float64").fillna(0)

        # General settings
        self.seasonPeriods = season_periods
        self.maxSamples = max_samples  # Time series
        self.differ = differ  # Correlations
        self.lags = lags  # Correlations

        # Storage settings
        self.mode = mode
        self.tag = pre_tag
        self.version = version if version is not None else 0
        self.skip = skip_completed
        self.folder = folder if folder == "" or folder[-1] == "/" else folder + "/"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # Create Base folder
        if not os.path.exists(self.folder + "EDA/"):
            os.mkdir(self.folder + "EDA")
        self.folder += "EDA/"

    def run(self):
        # Run all functions
        logger.info("Generating Missing Values Plot")
        self.missing_values()
        logger.info("Generating Time plots")
        self.time_plots()
        logger.info("Generating Box plots")
        self.box_plots()
        logger.info("Generating Co-linearity Plots")
        self.co_linearity()

        # Functions which require output
        if self.Y is not None:
            logger.info("Generating SHAP plot")
            self.shap()
            logger.info("Generating Feature Ranking Plot")
            self.feature_ranking()

        # Mode specific functions
        if self.mode == "classification":
            self._run_classification()
        else:
            self._run_regression()

    def _run_classification(self):
        pass

    def _run_regression(self):
        if self.plotSeasonality:
            logger.info("Generating Seasonality Plots")
            self.seasonality()
        logger.info("Generating Diff Var Plot")
        self.differencing()
        logger.info("Generating ACF Plots")
        self.complete_auto_corr()
        logger.info("Generating PACF Plots")
        self.partial_auto_corr()
        if self.Y is not None:
            logger.info("Generating CCF Plots")
            self.cross_corr()
            logger.info("Generating Scatter plots")
            self.scatters()

    def missing_values(self):
        if self.plotMissingValues:
            # Create folder
            if not os.path.exists(self.folder + "MissingValues/"):
                os.mkdir(self.folder + "MissingValues/")

            # Skip if exists
            if self.tag + "v{}.png".format(self.version) in os.listdir(
                self.folder + "MissingValues/"
            ):
                return

            # Plot
            fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [9, 1]})
            a0.imshow(
                self.data.isna(), cmap="Greys", aspect="auto", interpolation="none"
            )
            a0.set_ylabel("Rows")
            a0.set_xticks(range(len(self.data.keys())), self.data.keys())
            a1.plot(self.data.isna().sum(axis=1), range(len(self.data)), c="k")
            a1.axis("off")
            fig.savefig(self.folder + "MissingValues/v{}.png".format(self.version))

    def box_plots(self):
        if self.plotBoxPlots:
            # Create folder
            if not os.path.exists(self.folder + "BoxPlots/v{}/".format(self.version)):
                os.makedirs(self.folder + "BoxPlots/v{}/".format(self.version))

            # Iterate through vars
            for key in tqdm(self.data.keys()):

                # Skip if existing
                if self.tag + key + ".png" in os.listdir(
                    self.folder + "BoxPlots/v{}/".format(self.version)
                ):
                    continue

                # Figure prep
                fig = plt.figure(figsize=[24, 16])
                plt.title(key)

                # Classification
                if self.mode == "classification":
                    plt.boxplot(
                        [
                            self.data.loc[self.Y == 1, key],
                            self.data.loc[self.Y == -1, key],
                        ],
                        labels=["Faulty", "Healthy"],
                    )
                    plt.legend(["Faulty", "Healthy"])

                # Regression
                if self.mode == "regression":
                    plt.boxplot(self.data[key])

                # Store & Close
                fig.savefig(
                    self.folder
                    + "BoxPlots/v{}/".format(self.version)
                    + self.tag
                    + key
                    + ".png",
                    format="png",
                    dpi=300,
                )
                plt.close()

    def time_plots(self):
        if self.plotTimePlots:
            # Create folder
            if not os.path.exists(self.folder + "TimePlots/v{}/".format(self.version)):
                os.makedirs(self.folder + "TimePlots/v{}/".format(self.version))

            # Set pyplot limit
            matplotlib.use("Agg")
            matplotlib.rcParams["agg.path.chunksize"] = 200000

            # Under sample
            ind = np.linspace(0, len(self.data) - 1, self.maxSamples).astype("int")
            data, y = self.data.iloc[ind], self.Y.iloc[ind]

            # Iterate through features
            for key in tqdm(data.keys()):
                # Skip if existing
                if self.tag + key + ".png" in os.listdir(
                    self.folder + "TimePlots/v{}/".format(self.version)
                ):
                    continue

                # Figure preparation
                fig = plt.figure(figsize=[24, 16])
                plt.title(key)

                # Plot
                if self.mode == "classification":
                    cm = plt.get_cmap("bwr")
                else:
                    cm = plt.get_cmap("summer")
                nm_output = (y - y.min()) / (y.max() - y.min())
                plt.scatter(data.index, data[key], c=cm(nm_output), alpha=0.3)

                # Store & Close
                fig.savefig(
                    self.folder
                    + "TimePlots/v{}/".format(self.version)
                    + self.tag
                    + key
                    + ".png",
                    format="png",
                    dpi=100,
                )
                plt.close(fig)

    def seasonality(self):
        # Create folder
        if not os.path.exists(self.folder + "Seasonality/"):
            os.mkdir(self.folder + "Seasonality/")

        # Iterate through features
        for key in tqdm(self.data.keys()):
            if self.tag + key + "_v{}.png".format(self.version) in os.listdir(
                self.folder + "Seasonality/"
            ):
                continue
            seasonality = STL(self.data[key], period=self.seasonPeriods).fit()
            fig = plt.figure(figsize=[24, 16])
            plt.plot(range(len(self.data)), self.data[key])
            plt.plot(range(len(self.data)), seasonality)
            plt.title(key + ", period=" + str(self.seasonPeriods))
            fig.savefig(
                self.folder
                + "Seasonality/"
                + self.tag
                + str(self.seasonPeriods)
                + "/"
                + key
                + "_v{}.png".format(self.version),
                format="png",
                dpi=300,
            )
            plt.close()

    def co_linearity(self):
        if self.plotCoLinearity:
            # Create folder
            if not os.path.exists(
                self.folder + "CoLinearity/v{}/".format(self.version)
            ):
                os.makedirs(self.folder + "CoLinearity/v{}/".format(self.version))

            # Skip if existing
            if self.tag + "MinimumRepresentation.png" in os.listdir(
                self.folder + "CoLinearity/v{}/".format(self.version)
            ):
                return

            # Plot threshold matrix
            threshold = 0.95
            fig = plt.figure(figsize=[24, 16])
            plt.title("Co-linearity matrix, threshold {:.2f}".format(threshold))
            plt.imshow(abs(self.data.corr()) < threshold, cmap="gray")
            fig.savefig(
                self.folder
                + "CoLinearity/v{}/".format(self.version)
                + self.tag
                + "Matrix.png",
                format="png",
                dpi=300,
            )

            # Minimum representation
            corr_mat = self.data.corr().abs()
            upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype("bool"))
            col_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            minimal_rep = self.data.drop(self.data[col_drop], axis=1)
            fig = plt.figure(figsize=[24, 16])
            plt.imshow(abs(minimal_rep.corr()) < threshold, cmap="gray")
            fig.savefig(
                self.folder
                + "CoLinearity/v{}/".format(self.version)
                + self.tag
                + "Minimum_Representation.png",
                format="png",
                dpi=300,
            )

    def differencing(self):
        if self.plotDifferencing:
            # Create folder
            if not os.path.exists(self.folder + "Lags/"):
                os.mkdir(self.folder + "Lags/")

            # Skip if existing
            if self.tag + "Variance.png" in os.listdir(self.folder + "Lags/"):
                return

            # Setup
            max_lags = 4
            var_vec = np.zeros((max_lags, len(self.data.keys())))
            diff_data = self.data / np.sqrt(self.data.var())

            # Calculate variance per lag
            for i in range(max_lags):
                var_vec[i, :] = diff_data.var()
                diff_data = diff_data.diff(1)[1:]

            # Plot
            fig = plt.figure(figsize=[24, 16])
            plt.title("Variance for different lags")
            plt.plot(var_vec)
            plt.xlabel("Lag")
            plt.yscale("log")
            plt.ylabel("Average variance")
            fig.savefig(
                self.folder + "Lags/" + self.tag + "Variance.png", format="png", dpi=300
            )

    def complete_auto_corr(self):
        if self.plotSignalCorrelations:
            # Create folder
            if not os.path.exists(self.folder + "Correlation/ACF/"):
                os.makedirs(self.folder + "Correlation/ACF/")

            # Difference data
            diff_data = deepcopy(self.data)
            for i in range(self.differ):
                diff_data = diff_data.diff(1)[1:]

            # Iterate through features
            for key in tqdm(self.data.keys()):
                # Skip if existing
                if self.tag + key + "_differ_" + str(self.differ) + "_v{}.png".format(
                    self.version
                ) in os.listdir(self.folder + "Correlation/ACF/"):
                    continue

                # Plot
                fig = plot_acf(diff_data[key], fft=True)
                plt.title(key)
                fig.savefig(
                    self.folder
                    + "Correlation/ACF/"
                    + self.tag
                    + key
                    + "_differ_"
                    + str(self.differ)
                    + "_v{}.png".format(self.version),
                    format="png",
                    dpi=300,
                )
                plt.close()

    def partial_auto_corr(self):
        if self.plotSignalCorrelations:
            # Create folder
            if not os.path.exists(self.folder + "Correlation/PACF/"):
                os.makedirs(self.folder + "Correlation/PACF/")

            # Iterate through features
            for key in tqdm(self.data.keys()):
                # Skip if existing
                if self.tag + key + "_differ_" + str(self.differ) + "_v{}.png".format(
                    self.version
                ) in os.listdir(self.folder + "EDA/Correlation/PACF/"):
                    continue

                # Plot
                try:
                    fig = plot_pacf(self.data[key])
                    fig.savefig(
                        self.folder
                        + "EDA/Correlation/PACF/"
                        + self.tag
                        + key
                        + "_differ_"
                        + str(self.differ)
                        + "_v{}.png".format(self.version),
                        format="png",
                        dpi=300,
                    )
                    plt.title(key)
                    plt.close()
                except Exception as e:
                    # todo find exception
                    raise ValueError(e)

    def cross_corr(self):
        if self.plotSignalCorrelations:
            # Create folder
            if not os.path.exists(self.folder + "Correlation/Cross/"):
                os.makedirs(self.folder + "Correlation/Cross/")

            # Prepare
            folder = "Correlation/Cross/"
            y = self.Y.to_numpy().reshape((-1))

            # Iterate through features
            for key in tqdm(self.data.keys()):
                # Skip if existing
                if self.tag + key + "_differ_" + str(self.differ) + "_v{}.png".format(
                    self.version
                ) in os.listdir(self.folder + folder):
                    continue

                # Plot
                try:
                    fig = plt.figure(figsize=[24, 16])
                    plt.xcorr(self.data[key], y, maxlags=self.lags)
                    plt.title(key)
                    fig.savefig(
                        self.folder
                        + folder
                        + self.tag
                        + key
                        + "_differ_"
                        + str(self.differ)
                        + "_v{}.png".format(self.version),
                        format="png",
                        dpi=300,
                    )
                    plt.close()
                except Exception as e:
                    raise ValueError(e)

    def scatters(self):
        if self.plotScatterPlots:
            # Create folder
            if not os.path.exists(self.folder + "Scatters/v{}/".format(self.version)):
                os.makedirs(self.folder + "Scatters/v{}/".format(self.version))

            # Iterate through features
            for key in tqdm(self.data.keys()):
                # Skip if existing
                if "{}{}.png".format(self.tag, key) in os.listdir(
                    self.folder + "Scatters/v{}/".format(self.version)
                ):
                    continue

                # Plot
                fig = plt.figure(figsize=[24, 16])
                plt.scatter(self.Y, self.data[key], alpha=0.2)
                plt.ylabel(key)
                plt.xlabel("Output")
                plt.title("Scatter Plot " + key + " - Output")
                fig.savefig(
                    self.folder
                    + "Scatters/v{}/".format(self.version)
                    + self.tag
                    + key
                    + ".png",
                    format="png",
                    dpi=100,
                )
                plt.close(fig)

    def shap(self, args=None):
        args = args if args is not None else {}
        if self.plotFeatureImportance:
            # Create folder
            if not os.path.exists(self.folder + "Features/v{}/".format(self.version)):
                os.makedirs(self.folder + "Features/v{}/".format(self.version))

            # Skip if existing
            if self.tag + "SHAP.png" in os.listdir(
                self.folder + "Features/v{}/".format(self.version)
            ):
                return

            # Create model
            if self.mode == "classification":
                model = RandomForestClassifier(**args).fit(self.data, self.Y)
            else:
                model = RandomForestRegressor(**args).fit(self.data, self.Y)

            # Calculate SHAP values
            import shap

            shap_values = shap.TreeExplainer(model).shap_values(self.data)

            # Plot
            fig = plt.figure(figsize=[8, 32])
            plt.subplots_adjust(left=0.4)
            shap.summary_plot(shap_values, self.data, plot_type="bar")
            fig.savefig(
                self.folder
                + "Features/v{}/".format(self.version)
                + self.tag
                + "SHAP.png",
                format="png",
                dpi=300,
            )

    def feature_ranking(self, **args):
        args = args if args is not None else {}
        if self.plotFeatureImportance:
            # Create folder
            if not os.path.exists(self.folder + "Features/v{}/".format(self.version)):
                os.mkdir(self.folder + "Features/v{}/".format(self.version))

            # Skip if existing
            if self.tag + "RF.png" in os.listdir(
                self.folder + "Features/v{}/".format(self.version)
            ):
                return

            # Create model
            if self.mode == "classification":
                model = RandomForestClassifier(**args).fit(self.data, self.Y)
            else:
                model = RandomForestRegressor(**args).fit(self.data, self.Y)

            # Plot
            fig, ax = plt.subplots(figsize=[4, 6], constrained_layout=True)
            plt.subplots_adjust(left=0.5, top=1, bottom=0)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ind = np.argsort(model.feature_importances_)
            plt.barh(
                list(self.data.keys()[ind])[-15:],
                width=model.feature_importances_[ind][-15:],
                color="#2369ec",
            )
            fig.savefig(
                self.folder
                + "Features/v{}/".format(self.version)
                + self.tag
                + "RF.png",
                format="png",
                dpi=300,
            )
            plt.close()

            # Store results
            results = pd.DataFrame(
                {"x": self.data.keys(), "score": model.feature_importances_}
            )
            results.to_csv(
                self.folder + "Features/v{}/".format(self.version) + self.tag + "RF.csv"
            )

    # def predictive_power_score(self):
    #     if self.plotFeatureImportance:
    #         # Create folder
    #         if not os.path.exists(self.folder + "Features/v{}/".format(self.version)):
    #             os.mkdir(self.folder + "Features/v{}/".format(self.version))
    #
    #         # Skip if existing
    #         if self.tag + "PPScore.png" in os.listdir(self.folder + "Features/v{}/".format(self.version)):
    #             return
    #
    #         # Calculate PPS
    #         data = self.data.copy()
    #         if isinstance(self.Y, pd.core.series.Series):
    #             data.loc[:, "target"] = self.Y
    #         elif isinstance(self.Y, pd.DataFrame):
    #             data.loc[:, "target"] = self.Y.loc[:, self.Y.keys()[0]]
    #         pp_score = ppscore.predictors(data, "target").sort_values("ppscore")
    #
    #         # Plot
    #         fig, ax = plt.subplots(figsize=[4, 6])
    #         plt.subplots_adjust(left=0.5, top=1, bottom=0)
    #         ax.spines["right"].set_visible(False)
    #         ax.spines["bottom"].set_visible(False)
    #         ax.spines["top"].set_visible(False)
    #         plt.barh(pp_score["x"][-15:], width=pp_score["ppscore"][-15:], color="#2369ec")
    #         fig.savefig(self.folder + "Features/v{}/".format(self.version) + self.tag + "Ppscore.png", format="png",
    #                     dpi=400)
    #         plt.close()
    #
    #         # Store results
    #         pp_score.to_csv(self.folder + "Features/v{}/pp_score.csv".format(self.version))
