from typing import Self, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from dataclasses import dataclass
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from ..exceptions.custom_errors import PredictionsMissing, PredictionNotFound
from ..types.types import RandomForestModelTypes, RandomForestModelParams


@dataclass
class Prediction:
    observed: pd.DataFrame
    predicted: pd.Series
    _truth: Optional[pd.Series] = None

    def __post_init__(self):
        self.truth_known = self._truth

    @property
    def truth(self):
        return self._truth

    @truth.setter
    def truth(self, value):
        self._truth = value


RandomForestModelDict = {
    "classifier": RandomForestClassifier,
    "regressor": RandomForestRegressor,
}


class RandomForestModelTask:

    def __init__(self, model_type: RandomForestModelTypes = "classifier"):
        self.model_class = RandomForestModelDict[model_type]
        print(self.model_class)
        self.predictions: dict[str, Prediction] = {}

    def _has_model(self):
        if hasattr(self, "model"):
            return True

    def _handle_has_not_model(self):
        print(
            f"No model has been created yet. \
                Creating a {self.model_type} model with default parameters..."
        )
        self.create_model()

    def _handle_missing_predictions(self):
        raise PredictionsMissing(self.model)

    def get_prediction(self, prediction_name: str) -> Prediction:
        return self.predictions[prediction_name]

    @property
    def last_prediction(self):
        if self.predictions:
            return list(self.predictions.values())[-1]
        self._handle_missing_predictions()

    def feed_truth(self, prediction_name: str, truth: pd.Series) -> bool:
        if not self.predictions:
            self._handle_missing_predictions()
        try:
            prediction = self.predictions[prediction_name]
        except KeyError:
            PredictionNotFound(prediction_name)
        if len(truth) != len(prediction.predicted):
            raise ValueError(
                "Length of truth must be equal to length of prediction.\
                             All predictions must have a truth value to be valued against.\
                             Please check your data."
            )
        prediction.truth = truth
        self.predictions.update({prediction_name: prediction})
        return True

    def create_model(
        self, model_params: Optional[RandomForestModelParams] = None
    ) -> Self:
        if model_params is None:
            # Default parameters
            model_params = {
                "n_estimators": 100,
                "criterion": "gini",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "bootstrap": True,
                "oob_score": False,
                "n_jobs": None,
                "random_state": None,
                "verbose": 0,
                "warm_start": False,
                "class_weight": None,
                "ccp_alpha": 0.0,
                "max_samples": None,
            }
        self.model = self.model_class(**model_params)
        return self

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        split_threshold: float = 0.2,
        random_state: int = 42,
    ) -> Self:
        if not self._has_model():
            self._handle_has_not_model()
        self.feature_names = X.columns
        self.target_name = y.name
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_threshold, random_state=random_state
        )
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model.fit(X_train, y_train)
        print("Model has been fitted successfully.")
        return self

    def predict(self, prediction_name: Optional[str] = None) -> Prediction:
        """Predict on the test set split-out when fitting the model"""
        if not self._has_model():
            self._handle_has_not_model()
        check_is_fitted(self.model)
        X_observed = self.X_test
        prediction = Prediction(
            observed=X_observed, predicted=self.model.predict(X_observed)
        )
        prediction_name = prediction_name or str(int(time() * 1000))
        self.predictions[prediction_name] = Prediction(
            observed=X_observed, predicted=self.model.predict(X_observed)
        )
        return prediction

    def predict_on_new_observation(
        self, X_observed: pd.DataFrame, prediction_name: Optional[str]
    ) -> Prediction:
        if X_observed is None:
            raise ValueError("Provide observations you want to predict on.")
        if not self._has_model():
            self._handle_has_not_model()
        check_is_fitted(self.model)
        # Default name is current timestamp (int to remove milliseconds)
        prediction_name = prediction_name or str(int(time() * 1000))
        prediction = Prediction(
            observed=X_observed, predicted=self.model.predict(X_observed)
        )
        self.predictions.update({prediction_name: prediction})
        return prediction

    def get_classification_report(self, prediction_name: str) -> None:
        if not self.predictions:
            self._handle_missing_predictions()
        if prediction_name is None:
            candidate_pred = self.last_prediction
        else:
            candidate_pred = self.get_prediction(prediction_name)
        if not candidate_pred.truth_known:
            raise ValueError("No true values have been fed to this prediction.")
        return classification_report(
            candidate_pred.truth, candidate_pred.predicted, output_dict=True
        )

    def get_feature_importances(self, sorted: bool = True) -> pd.DataFrame:
        if not self._has_model():
            self._handle_has_not_model()
        check_is_fitted(self.model)
        feature_importances = pd.DataFrame(
            {"value": self.model.feature_importances_}, index=self.feature_names
        )
        if sorted:
            return feature_importances.sort_values("value", ascending=False)
        return feature_importances

    def get_decision_path(self, X):
        pass

    def plot_feature_importances(
        self,
        sorted: bool = True,
        figure_parameters: dict = {},
        axes_parameters: dict = {},
        **kwargs: dict,
    ) -> None:
        self._has_model()
        check_is_fitted(self.model)
        # TODO Plot feature importances here
        fig, ax = plt.subplots(**figure_parameters)
        feature_importances = self.get_feature_importances(sorted=sorted)
        index = np.arange(len(feature_importances))
        ax.barh(
            index, feature_importances["value"], color="purple", label="Random Forest"
        )
        ax.set(yticks=index, yticklabels=feature_importances.index, **axes_parameters)
        ax.legend()
        plt.show()
