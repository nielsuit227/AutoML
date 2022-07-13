#  Copyright (c) 2022 by Amplo.

import copy
import math
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold

from amplo.documenting.binary import BinaryDocumenting
from amplo.utils.logging import logger
from amplo.utils.util import deprecated

if TYPE_CHECKING:
    from amplo import Pipeline


@deprecated(
    "This is no longer used, all information is available in the created settings.json."
)
class RegressionDocumenting(BinaryDocumenting):
    def __init__(self, pipeline: "Pipeline"):
        super().__init__(pipeline)
        self.y_not_standardized = None

    def analyse(self):
        # Cross-Validation Plots
        fig, ax = plt.subplots(
            math.ceil(self.p.cv_splits / 2),
            2,
            sharex="all",
            sharey="all",
            figsize=[24, 8],
        )
        fig.suptitle(
            "{}-Fold Cross Validated Predictions - {}".format(
                self.p.cv_splits, self.mName
            )
        )

        # Initialize iterables
        mae = []
        mse = []
        r2 = []
        max_error = []
        rel_error = []
        self.cv = KFold(n_splits=self.p.cv_splits, shuffle=self.p.shuffle)
        # Cross Validate
        i = 0
        for i, (t, v) in enumerate(self.cv.split(self.x, self.y)):
            xt, xv, yt, yv = (
                self.x[t],
                self.x[v],
                self.y[t].reshape((-1)),
                self.y[v].reshape((-1)),
            )
            model = copy.deepcopy(self.model)
            model.fit(xt, yt)
            prediction = model.predict(xv)

            # Metrics
            mae.append(metrics.mean_absolute_error(yv, prediction))
            mse.append(metrics.mean_squared_error(yv, prediction))
            r2.append(metrics.r2_score(yv, prediction))
            max_error.append(metrics.max_error(yv, prediction))
            rel_error.append(metrics.mean_absolute_percentage_error(yv, prediction))

            # Plot
            ax[i // 2][i % 2].set_title("Fold-{}".format(i))
            if self.p.standardize:
                ax[i // 2][i % 2].plot(self.y_not_standardized[v], color="#2369ec")
                ax[i // 2][i % 2].plot(
                    self.p.bestOutputScaler.inverse_transform(model.predict(xv)),
                    color="#ffa62b",
                    alpha=0.4,
                )
            else:
                ax[i // 2][i % 2].plot(yv, color="#2369ec")
                ax[i // 2][i % 2].plot(model.predict(xv), color="#ffa62b", alpha=0.4)

        # Store
        self.metrics = {
            "Mean Absolute Error": [float(np.mean(mae)), float(np.std(mae))],
            "Mean Squared Error": [float(np.mean(mse)), float(np.std(mse))],
            "R2 Score": [float(np.mean(r2)), float(np.std(r2))],
            "Max Error": [float(np.mean(max_error)), float(np.std(max_error))],
            "Mean Relative Error": [
                float(np.mean(rel_error)),
                float(np.std(rel_error)),
            ],
        }
        self.outputMetrics = copy.deepcopy(self.metrics)

        # Save figure
        ax[i // 2][i % 2].legend(["Output", "Prediction"])
        if not os.path.exists(
            self.p.main_dir + "EDA/Validation/v{}/".format(self.p.version)
        ):
            os.makedirs(self.p.main_dir + "EDA/Validation/v{}/".format(self.p.version))
        cross_val_path = self.p.main_dir + "EDA/Validation/v{}/Cross_Val_{}.png".format(
            self.p.version, self.mName
        )
        fig.savefig(cross_val_path, format="png", dpi=200)

        # Print & Finish plot
        logger.info(
            "Mean Absolute Error:            {:.2f} \u00B1 {:.2f}".format(
                np.mean(mae), np.std(mae)
            )
        )
        logger.info(
            "Mean Squared Error:             {:.2f} \u00B1 {:.2f}".format(
                np.mean(mse), np.std(mse)
            )
        )
        logger.info(
            "R2 Score:                       {:.2f} \u00B1 {:.2f}".format(
                np.mean(r2), np.std(r2)
            )
        )
        logger.info(
            "Max Error:                      {:.2f} \u00B1 {:.2f}".format(
                np.mean(max_error), np.std(max_error)
            )
        )
        logger.info(
            "Mean Absolute Percentage Error: {:.2f} \u00B1 {:.2f}".format(
                np.mean(rel_error), np.std(rel_error)
            )
        )

        # Feature Importance
        if not os.path.exists(
            self.p.main_dir + "EDA/Features/v{}/RF.png".format(self.p.version)
        ):
            if not os.path.exists(
                self.p.main_dir + "EDA/Features/v{}".format(self.p.version)
            ):
                os.makedirs(
                    self.p.main_dir + "EDA/Features/v{}/".format(self.p.version)
                )
            fig, ax = plt.subplots(figsize=[4, 6], constrained_layout=True)
            plt.subplots_adjust(left=0.5, top=1, bottom=0)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            fi = self.p.feature_processor.feature_importance_["rf"]
            plt.barh(list(fi)[:15], width=list(fi.values())[:15], color="#2369ec")
            fig.savefig(
                self.p.main_dir + "EDA/Features/v{}/RF.png".format(self.p.version),
                format="png",
                dpi=200,
            )

    def model_performance(self):
        if not self.check_new_page():
            self.ln(self.lh)
        self.add_h2("Model Performance")
        self.add_text(
            "Model performance is analysed by various metrics. This model has been "
            "selected based on the {} score.".format(self.p.objective)
        )

        # Metrics
        self.set_font("Helvetica", "B", 12)
        self.ln(self.lh)
        self.cell(w=50, h=self.lh, txt="Metric", border="B", align="C")
        self.cell(w=50, h=self.lh, txt="Score", border="LB", align="C")
        self.set_font("Helvetica", "", 12)
        for k, v in self.metrics.items():
            self.ln(self.lh)
            self.cell(w=50, h=self.lh, txt=k, border="R", align="L")
            self.cell(
                w=50,
                h=self.lh,
                txt=f"{v[0]:.2f} \u00B1 {v[1]:.2f}",
                border="L",
                align="C",
            )
        self.ln(self.lh * 3)

    def validation(self):
        if not self.check_new_page():
            self.ln(self.lh)

        self.add_h3("Cross Validation Plot")
        x, y = self.get_x(), self.get_y()
        path = self.p.main_dir + "EDA/Validation/v{}/Cross_Val_{}.png".format(
            self.p.version, self.mName
        )
        self.image(x=(self.WIDTH - 200) / 2, y=y, w=200, h=90, name=path)
        self.add_page()

        self.add_h3("Validation Strategy")
        self.add_text(
            "All experiments are cross-validated. This means that every time a model's "
            "performance is evaluated, it's trained on one part of the data, and test "
            "on another. Therefore, the model is always test against data it has not "
            "yet been trained for. This gives the best approximation for real world "
            "(out of sample) performance. The current validation strategy used "
            "is {}, with {} splits and {} shuffling the data.".format(
                type(self.cv).__name__,
                self.p.cv_splits,
                "with" if self.p.shuffle else "without",
            )
        )
