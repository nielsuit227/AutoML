from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from amplo.automl.feature_processing.feature_selection import FeatureSelector


class TestFeatureSelector:
    def test_fit(self, classification_data):
        selector = FeatureSelector("target", mode="classification")
        selector.fit(classification_data)
        rf_fi = selector.feature_importance_["rf"]
        shap_fi = selector.feature_importance_["shap"]

        assert "rf_increment" in selector.feature_sets_
        assert "rf_threshold" in selector.feature_sets_
        assert "shap_increment" in selector.feature_sets_
        assert "shap_threshold" in selector.feature_sets_
        for key in classification_data:
            if key == "target":
                continue
            assert key in rf_fi
            assert key in shap_fi
            assert key in selector.all_features

        y_values = classification_data["target"].values.reshape(-1, 1)
        for fi in [rf_fi, shap_fi]:
            max_values = classification_data[max(fi, key=lambda x: fi[x])]
            min_values = classification_data[min(fi, key=lambda x: fi[x])]
            max_score = cross_val_score(
                DecisionTreeClassifier(),
                max_values.values.reshape(-1, 1),
                y_values,
                cv=3,
                scoring="neg_log_loss",
            )
            min_score = cross_val_score(
                DecisionTreeClassifier(),
                min_values.values.reshape(-1, 1),
                y_values,
                cv=3,
                scoring="neg_log_loss",
            )
            assert sum(max_score) > sum(min_score)
