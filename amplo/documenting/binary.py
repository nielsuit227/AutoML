#  Copyright (c) 2022 by Amplo.

import copy
import math
import os
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.error import URLError

import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.model_selection import StratifiedKFold

from amplo.automl.feature_processing import (
    NopFeatureExtractor,
    StaticFeatureExtractor,
    TemporalFeatureExtractor,
)
from amplo.utils.logging import logger
from amplo.utils.util import deprecated

if TYPE_CHECKING:
    from amplo import Pipeline


@deprecated(
    "This is no longer used, all information is available in the created settings.json."
)
class BinaryDocumenting(FPDF):
    def __init__(self, pipeline: "Pipeline"):
        super().__init__()

        # Settings
        self.WIDTH = 210
        self.HEIGHT = 297
        self.lh = 6  # Line Height
        self.pm = 15  # Paper margin
        self.tm = 8  # Text Margins
        self.set_margins(self.pm, self.pm, self.pm)

        # Args
        self.p = pipeline
        self.name = pipeline.name

        # Initiates
        self.model = None
        self.mName = None
        self.feature_set = None
        self.metrics = None
        self.outputMetrics = None
        self.confusion_matrix = None
        self.cv = None
        self.x = None
        self.y = None
        self.featureImportances = {}

    def header(self):
        try:  # No problem if not possible due lack of internet
            self.image(
                "https://raw.githubusercontent.com/nielsuit227/AutoML/main/Amplo/Static/logo.png",
                x=self.pm,
                y=self.pm,
                w=55,
                h=15,
            )
        except URLError:
            pass
        self.set_font("Helvetica", "", 14)
        self.set_text_color(100, 100, 100)
        self.cell(self.WIDTH - 80)
        self.cell(0, 15, "Amplo AutoML Documentation", 0, 0, align="R")
        self.set_font("Helvetica", "", 10)
        self.ln(6)
        self.cell(
            0,
            15,
            datetime.now().strftime("%d %b %Y - %H:%M"),
            0,
            0,
            align="R",
        )
        self.ln(25)

    def create(self, model, feature_set):
        """
        Creates the entire documentation.

        Parameters
        ----------
        model : Model to document / validate
        feature_set : Feature set to document / validate on
        """
        # Asserts
        assert model is not None, "Model cannot be none."
        assert feature_set in self.p.feature_sets.keys(), "Feature set unavailable"

        # Set variables
        self.model = model
        self.mName = type(model).__name__
        self.feature_set = feature_set

        # Analyse the model
        self.prepare_data()
        self.analyse()

        # Check if folder needs creation
        path = self.p.main_dir + "Documentation/v{}/{}_{}.pdf".format(
            self.p.version, self.mName, feature_set
        )
        if not os.path.exists(path[: path.rfind("/")]):
            os.makedirs(path[: path.rfind("/")])

        # Create PDF
        self.add_page()
        self.add_h1("{}".format(self.name))
        self.add_h2("{} - v{}".format(self.mName, self.p.version))
        self.add_text(self.model_description)
        self.model_performance()
        self.validation()
        self.parameters()
        self.features()
        self.data()
        self.score_board()
        self.output(
            self.p.main_dir
            + "Documentation/v{}/{}_{}.pdf".format(
                self.p.version, self.mName, self.feature_set
            )
        )

    def prepare_data(self):
        """
        Although data of the Pipeline (self.p.x and self.p.y), is already cleaned, it
        is not normalized or sequenced.
        @return:
        @rtype:
        """
        x, y = self.p.x, self.p.y
        self.x, self.y = x.to_numpy(), y.to_numpy()

    def analyse(self):
        # Metrics & Confusion Matrix
        accuracy = []
        precision = []
        sensitivity = []
        specificity = []
        f1_score = []
        area_under_curves = []
        true_positive_rates = []
        cm = np.zeros((self.p.cv_splits, 2, 2))
        mean_fpr = np.linspace(0, 1, 100)

        # Plot Initiator
        fig, ax = plt.subplots(
            math.ceil(self.p.cv_splits / 2),
            2,
            sharex="all",
            sharey="all",
            figsize=[12, 8],
        )
        fig.suptitle(
            "{}-Fold Cross Validated Predictions - {} ({})".format(
                self.p.cv_splits, self.mName, self.feature_set
            )
        )
        fig2, ax2 = plt.subplots(figsize=[12, 8])

        # Modelling
        self.cv = StratifiedKFold(n_splits=self.p.cv_splits, shuffle=self.p.shuffle)
        for i, (t, v) in enumerate(self.cv.split(self.x, self.y)):
            n = len(v)
            xt, xv, yt, yv = (
                self.x[t],
                self.x[v],
                self.y[t].reshape((-1)),
                self.y[v].reshape((-1)),
            )
            model = copy.deepcopy(self.model)
            model.fit(xt, yt)
            predictions = model.predict(xv).reshape((-1))

            # Metrics
            tp = np.logical_and(np.sign(predictions) == 1, yv == 1).sum()
            tn = np.logical_and(np.sign(predictions) == 0, yv == 0).sum()
            fp = np.logical_and(np.sign(predictions) == 1, yv == 0).sum()
            fn = np.logical_and(np.sign(predictions) == 0, yv == 1).sum()
            accuracy.append((tp + tn) / n * 100)
            if tp + fp > 0:
                precision.append(tp / (tp + fp) * 100)
            if tp + fn > 0:
                sensitivity.append(tp / (tp + fn) * 100)
            if tn + fp > 0:
                specificity.append(tn / (tn + fp) * 100)
            if tp + fp > 0 and tp + fn > 0:
                f1_score.append(
                    2
                    * precision[-1]
                    * sensitivity[-1]
                    / (precision[-1] + sensitivity[-1])
                    if precision[-1] + sensitivity[-1] > 0
                    else 0
                )
            cm[i] = np.array([[tp, fp], [fn, tn]])

            # ROC calculations
            viz = RocCurveDisplay.from_predictions(
                yv, predictions, name="ROC fold {}".format(i + 1), ax=ax2
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            true_positive_rates.append(interp_tpr)
            area_under_curves.append(viz.roc_auc)

            # Plot
            ax[i // 2][i % 2].plot(yv, c="#2369ec", alpha=0.6)
            ax[i // 2][i % 2].plot(predictions, c="#ffa62b")
            ax[i // 2][i % 2].set_title("Fold-{}".format(i))

        # Statistics on results
        totals = np.sum(cm, axis=(1, 2), keepdims=True)
        means = np.mean(cm / totals * 100, axis=0).astype("float")
        stds = np.std(cm / totals * 100, axis=0).astype("float")

        # Store
        self.metrics = {
            "Accuracy": [float(np.mean(accuracy)), float(np.std(accuracy))],
            "Precision": [float(np.mean(precision)), float(np.std(precision))],
            "Sensitivity": [float(np.mean(sensitivity)), float(np.std(sensitivity))],
            "Specificity": [float(np.mean(specificity)), float(np.std(specificity))],
            "F1 Score": [float(np.mean(f1_score)), float(np.std(f1_score))],
        }
        self.confusion_matrix = {
            "means": means,
            "stds": stds,
        }
        self.outputMetrics = copy.deepcopy(self.metrics)
        self.outputMetrics["True Positives"] = [
            means[0, 0],
            stds[0, 0],
        ]  # These are float32s
        self.outputMetrics["False Positives"] = [means[0, 1], stds[0, 1]]
        self.outputMetrics["False Negatives"] = [means[1, 0], stds[1, 0]]
        self.outputMetrics["True Negatives"] = [means[1, 1], stds[1, 1]]

        # Print Results
        logger.info(
            "[AutoML] Accuracy:        {:.2f} \u00B1 {:.2f} %".format(
                np.mean(accuracy), np.std(accuracy)
            )
        )
        logger.info(
            "[AutoML] Precision:       {:.2f} \u00B1 {:.2f} %".format(
                np.mean(precision), np.std(precision)
            )
        )
        logger.info(
            "[AutoML] Sensitivity:     {:.2f} \u00B1 {:.2f} %".format(
                np.mean(sensitivity), np.std(sensitivity)
            )
        )
        logger.info(
            "[AutoML] Specificity:     {:.2f} \u00B1 {:.2f} %".format(
                np.mean(specificity), np.std(specificity)
            )
        )
        logger.info(
            "[AutoML] F1-score:        {:.2f} \u00B1 {:.2f} %".format(
                np.mean(f1_score), np.std(f1_score)
            )
        )
        logger.info("[AutoML] Confusion Matrix:")
        logger.info("[AutoML] Prediction / true |     Faulty     |    Healthy      ")
        logger.info(
            "[AutoML]       Faulty      | {} |  {}".format(
                ("{:.1f} \u00B1 {:.1f} %".format(means[0, 0], stds[0, 0])).ljust(14),
                ("{:.1f} \u00B1 {:.1f} %".format(means[0, 1], stds[0, 1])).ljust(14),
            )
        )
        logger.info(
            "[AutoML]       Healthy     | {} |  {}".format(
                ("{:.1f} \u00B1 {:.1f} %".format(means[1, 0], stds[1, 0])).ljust(14),
                ("{:.1f} \u00B1 {:.1f} %".format(means[1, 1], stds[1, 1])).ljust(14),
            )
        )

        # Check whether plot is possible
        if type(self.model).__name__ == "Lasso" or "Ridge" in type(self.model).__name__:
            return

        # Adjust plots
        ax2.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            lw=2,
            color="#ffa62b",
            label="Chance",
            alpha=0.8,
        )
        mean_tpr = np.mean(true_positive_rates, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(area_under_curves)
        ax2.plot(
            mean_fpr,
            mean_tpr,
            color="#2369ec",
            label=r"Mean ROC (AUC = {:.2f} $\pm$ {:.2f})".format(mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )
        std_tpr = np.std(true_positive_rates, axis=0)
        true_pos_rates_upper = np.minimum(mean_tpr + std_tpr, 1)
        true_pos_rates_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax2.fill_between(
            mean_fpr,
            true_pos_rates_lower,
            true_pos_rates_upper,
            color="#729ce9",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        ax2.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title="ROC Curve - {}".format(self.mName),
        )
        ax2.legend(loc="lower right")
        if not os.path.exists(
            self.p.main_dir + "EDA/Validation/v{}/".format(self.p.version)
        ):
            os.makedirs(self.p.main_dir + "EDA/Validation/v{}/".format(self.p.version))
        roc_path = self.p.main_dir + "EDA/Validation/v{}/ROC_{}.png".format(
            self.p.version, self.mName
        )
        fig2.savefig(roc_path, format="png", dpi=200)
        cross_val_path = self.p.main_dir + "EDA/Validation/v{}/Cross_Val_{}.png".format(
            self.p.version, self.mName
        )
        fig.savefig(cross_val_path, format="png", dpi=200)

        # Feature Importance (only if EDA is not run)
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

    def check_new_page(self, margin=220):
        if self.get_y() > margin:
            self.add_page()
            return True
        else:
            return False

    def add_h1(self, title):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(80, 80, 80)
        self.cell(0, 0, title, 0, 0)
        self.ln(self.lh * 2)

    def add_h2(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(80, 80, 80)
        self.cell(0, 0, title, 0, 0)
        self.ln(self.lh)

    def add_h3(self, title, **args):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(80, 80, 80)
        self.cell(0, 0, title, **args)
        self.ln(self.lh)

    def add_text(self, text):
        self.set_font("Helvetica", "", 12)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 5, text)

    def add_image(self, path, height, width):
        self.image(path, h=height, w=width, x=self.WIDTH - width)

    @property
    def model_description(self):
        if "CatBoost" in self.mName:
            return (
                "CatBoost, or Categorical Boosting,  is an algorithm for gradient "
                "boosting on decision trees, with natural implementation for "
                "categorical variables. It is similar to XGBoost and LightGBM but "
                "differs in implementation of the optimization algorithm. We often "
                "see this algorithm performing very well."
            )
        elif "XGB" in self.mName:
            return (
                "XGBoost, or Extreme Gradient Boosting, is an algorithm for gradient "
                "boosting on decision trees. It trains many decision trees sequentially"
                ", the additional tree always trying to mitigate the error of the whole"
                " model. XGBoost was the first gradient boosting algorithm to be "
                "implemented and is currently widely adopted in the ML world. "
            )
        elif "LGBM" in self.mName:
            return (
                "LightGBM, or Light Gradient Boosting Machine, is an iteration on the "
                "XGBoost algorithm. Similarly, it uses gradient boosting with decision "
                "trees. However, XGBoost tend to be slow for a larger number of samples"
                " (>10.000), but with leaf-wise growth instead of depth-wise growth, "
                "LightGBM increases training speed significantly. Performance is often "
                "close to XGBoost."
            )
        elif "HistG" in self.mName:
            return (
                "SciKits implementation of LightGBM, or Light Gradient Boosting Machine"
                ", is an iteration on the XGBoost algorithm. Similarly, uses gradient"
                " boosting with decision trees. However, XGBoost tend to be slow for a"
                " larger number of samples (>10.000), but with a different growth "
                "algorithm, LightGBM increases training speed significantly."
                " Performance is often close to XGBoost."
            )
        elif "GradientB" in self.mName:
            return (
                "SciKits implementation of XGBoost, or Extreme Gradient Boosting, is an"
                " algorithm for gradient boosting on decision trees. It trains many "
                "decision trees sequentially, the additional tree always trying to "
                "mitigate the error of the whole model. XGBoost was the first gradient"
                " boosting algorithm to be implemented and is currently widely adopted"
                " in the ML world. "
            )
        elif "RandomForest" in self.mName:
            return (
                "Random Forest, is an ensemble algorithm that combines many (100-1000) decision trees and "
                "predicts the average of all trained trees. Though gradient boosting methods often outperform "
                "Random Forest, some data characteristics favor the Random Forests performance. "
            )
        elif "Linear" in self.mName:
            return (
                "Linear models are simple algorithms where the inputs are multiplied by optimized weights to "
                "predict the output. "
            )
        elif "Bagging" in self.mName:
            return (
                "Bagging algorithms is an ensemble algorithm. Just like a Random "
                "Forest, it trains many Decision Trees. It then makes a prediction "
                "based on a voting base, the average of the prediction of the "
                "individual Decision Trees will be predicted by the Bagging algorithm."
                "Contrary to a Random Forest, the Bagging algorithm does not allocate "
                "subsets of features or data to the individual Decision Trees. "
            )
        elif "Stacking" in self.mName:
            return (
                "A Stacking algorithm is a linear algorithm that uses the prediction of"
                " various models. In our AutoML pipeline, we first train all included "
                "models, optimize the hyper parameters of the well performing ones and "
                "then take the three best combinations. Additionally, we add a Naive "
                "Bayes, a Linear Model and a K-Nearest Neighbors algorithm on top."
            )
        else:
            return """Model description yet to be included in this documenter."""

    def model_performance(self):
        if not self.check_new_page():
            self.ln(self.lh)
        self.add_h2("Model Performance")
        self.add_text(
            "Model performance is analysed by various metrics. This model has been selected based on the {} "
            "score.".format(self.p.objective)
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
                txt="{:.2f} \u00B1 {:.2f} %".format(v[0], v[1]),
                border="L",
                align="C",
            )
        self.ln(self.lh * 3)

        # Confusion Matrix
        n_classes = 2
        cell_width = int(
            (self.WIDTH - self.pm * 4 - self.tm * n_classes) / (n_classes + 2)
        )
        self.add_h3("Confusion Matrix")

        # First row
        self.set_font("Helvetica", "B", 12)
        self.cell(w=cell_width * 2, h=self.lh, txt="", align="L", border="R")
        self.cell(w=cell_width * n_classes, h=self.lh, txt="True Class", align="C")
        self.ln(self.lh)

        # Second Row
        self.cell(w=cell_width * 2, h=self.lh, txt="", align="L", border="BR")
        self.cell(w=cell_width, h=self.lh, txt="Faulty", align="C", border="B")
        self.cell(w=cell_width, h=self.lh, txt="Healthy", align="C", border="B")
        self.ln(self.lh)

        # Third Row (first with values)
        self.cell(w=cell_width, h=self.lh * 2, txt="Prediction", align="L")
        self.cell(w=cell_width, h=self.lh, txt="Faulty", align="L", border="R")
        self.set_font("Helvetica", "", 12)
        self.cell(
            w=cell_width,
            h=self.lh,
            txt="{:.2f} \u00B1 {:.2f} %".format(
                self.confusion_matrix["means"][0][0],
                self.confusion_matrix["stds"][0][0],
            ),
            align="C",
        )
        self.cell(
            w=cell_width,
            h=self.lh,
            txt="{:.2f} \u00B1 {:.2f} %".format(
                self.confusion_matrix["means"][0][1],
                self.confusion_matrix["stds"][0][1],
            ),
            align="C",
        )
        self.ln(self.lh)

        # Fourth Row
        self.set_font("Helvetica", "B", 12)
        self.cell(w=cell_width, h=self.lh, txt="", align="L")
        self.cell(w=cell_width, h=self.lh, txt="Healthy", align="L", border="R")
        self.set_font("Helvetica", "", 12)
        self.cell(
            w=cell_width,
            h=self.lh,
            txt="{:.2f} \u00B1 {:.2f} %".format(
                self.confusion_matrix["means"][1][0],
                self.confusion_matrix["stds"][1][0],
            ),
            align="C",
        )
        self.cell(
            w=cell_width,
            h=self.lh,
            txt="{:.2f} \u00B1 {:.2f} %".format(
                self.confusion_matrix["means"][1][1],
                self.confusion_matrix["stds"][1][1],
            ),
            align="C",
        )
        self.ln(self.lh)

    def validation(self):
        if not self.check_new_page():
            self.ln(self.lh)
        self.ln(self.lh * 2)
        self.add_h3("Area Under Curve & Cross Validation Plots")
        x, y = self.get_x(), self.get_y()
        path = self.p.main_dir + "EDA/Validation/v{}/ROC_{}.png".format(
            self.p.version, self.mName
        )
        self.image(x=x, y=y, w=90, h=60, name=path)
        path = path[: path.rfind("/")] + "/Cross_Val_{}.png".format(self.mName)
        self.image(x=x + self.WIDTH / 2 - self.pm, y=y, w=90, h=60, name=path)
        self.ln(self.lh)
        self.add_page()
        self.add_h3("Validation Strategy")
        self.add_text(
            "All experiments are cross-validated. This means that every time a model's "
            "performance is evaluated, it's trained on one part of the data, and test on another. Therefore, "
            "the model is always test against data it has not yet been trained for. This gives the best "
            "approximation for real world (out of sample) performance. The current validation strategy used "
            "is {}, with {} splits and {} shuffling the data.".format(
                type(self.cv).__name__,
                self.p.cv_splits,
                "with" if self.p.shuffle else "without",
            )
        )
        self.ln(self.lh)

    def parameters(self):
        if not self.check_new_page():
            self.ln(self.lh)
        params = self.model.get_params()
        n_params = len(params)
        keys, values = list(params.keys()), list(params.values())
        self.ln(self.lh)
        self.add_h2("Model Parameters")
        self.set_font("Helvetica", "B", 12)

        # Double rows
        if n_params > 15:
            # First row
            w = 30
            self.cell(w=w + 30, h=self.lh, txt="Parameter", align="C", border="RB")
            self.cell(w=w, h=self.lh, txt="Value", align="C", border="B")
            self.cell(w=5, h=self.lh, txt="")
            self.cell(w=w + 30, h=self.lh, txt="Parameter", align="C", border="RB")
            self.cell(w=w, h=self.lh, txt="Value", align="C", border="B")
            self.set_font("Helvetica", "", 12)
            n_rows = -(-n_params // 2)
            for i in range(n_rows):
                self.ln(self.lh)
                self.cell(w=w + 30, h=self.lh, txt=keys[i * 2], align="L", border="R")
                value = (
                    "{:.4e}".format(values[i * 2])
                    if isinstance(values[i * 2], float)
                    else str(values[i * 2])
                )
                self.cell(w=w, h=self.lh, txt=value, align="C")
                self.cell(w=5, h=self.lh),
                if i * 2 + 1 < n_params:
                    self.cell(
                        w=w + 30, h=self.lh, txt=keys[i * 2 + 1], align="L", border="R"
                    )
                    value = (
                        "{:.4e}".format(values[i * 2 + 1])
                        if isinstance(values[i * 2 + 1], float)
                        else str(values[i * 2 + 1])
                    )
                    self.cell(w=w, h=self.lh, txt=value, align="C")

        # Single rows
        else:
            w = 50
            self.cell(w=w, h=self.lh, txt="Parameter", align="L", border="RB")
            self.cell(w=w, h=self.lh, txt="Value", align="L", border="B")
            for i in range(n_params):
                self.ln(self.lh)
                self.cell(w=w, h=self.lh, txt=keys[i], align="L", border="R")
                value = (
                    "{:.4e}".format(values[i])
                    if isinstance(values[i], float)
                    else str(values[i])
                )
                self.cell(w=w, h=self.lh, txt=value, align="R")

    def features(self):
        feature_extractor = self.p.feature_processor.feature_extractor
        if isinstance(feature_extractor, NopFeatureExtractor):
            features = {
                "Raw Features": feature_extractor.features_,
            }
        elif isinstance(feature_extractor, StaticFeatureExtractor):
            features = {
                "Raw Features": feature_extractor.raw_features_,
                "Arithmetic Features": feature_extractor.cross_features_,
                "K-Means Features": feature_extractor.k_means_features_,
                "Trigonometric Features": feature_extractor.trigo_features_,
                "Inverse Features": feature_extractor.inverse_features_,
            }
        elif isinstance(feature_extractor, TemporalFeatureExtractor):
            features = {
                "Raw Features": feature_extractor.raw_features_,
                "Wavelet Features": feature_extractor.wav_features_,
            }
        else:
            msg = (
                f"Feature extractor of type "
                f"{type(feature_extractor).__name__} not recognized."
            )
            raise TypeError(msg)
        features.update(
            {
                "Collinear Features": self.p.feature_processor.collinear_cols_,
            }
        )

        if not self.check_new_page():
            self.ln(20)
        self.add_h2("Features")

        # Feature Extraction
        self.add_h3("Feature Extraction")
        self.add_text(
            "First, features that are collinear (a * x = y) up to {} % were removed. "
            "This resulted in {} removed features: {}{}".format(
                self.p.information_threshold * 100,
                len(features["Collinear Features"]),
                ", ".join([i for i in features["Collinear Features"][:20]]),
                ", ..." if len(features["Collinear Features"]) > 20 else " ",
            )
        )
        self.check_new_page(margin=220)
        self.add_text(
            "Subsequently, the features were manipulated and analysed to extract "
            "additional information. All promising combinations are analysed with a "
            "single shallow decision tree."
        )
        features.pop("Collinear Features")
        self.ln()
        self.set_font("Helvetica", "B", 12)
        self.cell(w=50, h=self.lh, txt="Sort Feature", border="BR", align="C")
        self.cell(w=50, h=self.lh, txt="Quantity", border="B", align="C")
        # Not supported now as multi cell adds whitespace in between rows, maybe appendix?
        # self.cell(w=120, h=self.lh, txt='Features', border='B', align='C')
        self.set_font("Helvetica", "", 12)
        # n_rows = max([len(v) for v in features.values()])
        for k, v in features.items():
            self.ln(self.lh)
            # h = max(- (- sum([len(i) for i in v]) // 60), 1)
            self.cell(w=50, h=self.lh, txt=k, border="R", align="L")
            self.cell(w=50, h=self.lh, txt="{}".format(len(v)), align="C")
            # self.multi_cell(w=140, h=self.lh, txt=', '.join([i for i in v]), align='L')
        self.check_new_page()

        # Feature Selection
        self.add_h3("Feature Selection")
        self.set_font("Helvetica", "", 12)
        y = self.get_y()
        self.multi_cell(
            w=80,
            h=self.lh,
            txt="Using a Random Forest model, the non-linear Feature Importance is analysed. The feature "
            "importance is measured in Mean Decrease in Gini Impurity. The feature importance is used "
            "to create two feature sets, one that contains 85% of all feature importance, and one that "
            "contains all features that contribute more than 1% to the total feature importance.\nThe "
            "top 15 feature with their mean decrease in Gini impurity are visualized on the right.",
        )
        path = self.p.main_dir + "EDA/Features/v{}/RF.png".format(self.p.version)
        if os.path.exists(path):
            self.image(name=path, x=110 + self.pm, y=y - 10, w=80, h=120)

    def data(self):
        if not self.check_new_page():
            self.ln(30)
        self.add_h2("Data Processing")
        self.add_text(
            "The following data manipulations were made to clean the data:\n"
            "  1.  Removed {} duplicate columns and {} duplicate rows\n"
            "  2.  Removed {} outliers with {}\n"
            "  3.  Imputed {} missing values with {}\n"
            "  4.  Removed {} columns with constant values\n".format(
                self.p.data_processor.removedDuplicateColumns,
                self.p.data_processor.removedDuplicateRows,
                self.p.data_processor.removedOutliers,
                self.p.data_processor.outlier_removal,
                self.p.data_processor.imputedMissingValues,
                self.p.data_processor.missing_values,
                self.p.data_processor.removedConstantColumns,
            )
        )

    def score_board(self):
        if self.p.results is None:
            return
        if not self.check_new_page():
            self.ln(self.lh)
        scores = self.p.sort_results(self.p.results)
        self.ln(self.lh)
        self.add_h2("Model Score Board")
        self.add_text(
            "Not only the {} has been optimized by the AutoML pipeline. In total, {} models where trained. "
            "The following table shows the performance of the top 10 performing models:".format(
                self.mName, len(self.p.results)
            )
        )
        self.ln(self.lh)
        # Table
        self.set_font("Helvetica", "B", 12)
        self.cell(w=70, h=self.lh, txt="Model", border="RB", align="C")
        self.cell(w=70, h=self.lh, txt=self.p.objective, border="B", align="C")
        # no space for parameters, move to appendix if wanted
        # self.cell(w=100, h=self.lh, txt='Parameters', border='B', align='C')
        self.set_font("Helvetica", "", 12)
        for i in range(min(10, len(scores))):
            self.ln(self.lh)
            self.cell(
                w=70,
                h=self.lh,
                txt="{}".format(scores.iloc[i]["model"]),
                border="R",
                align="C",
            )
            self.cell(
                w=70,
                h=self.lh,
                txt="{:.4f} \u00B1 {:.4f} %".format(
                    scores.iloc[i]["mean_objective"], scores.iloc[i]["std_objective"]
                ),
                align="C",
            )
