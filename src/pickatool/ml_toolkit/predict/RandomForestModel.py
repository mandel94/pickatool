from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Literal, Self, TypedDict, Optional, Union
from enum import Enum
from dataclasses import dataclass
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


RandomForestModelParams = TypedDict(
    "RandomForestModelParams",
    {
        "n_estimators": int,
        "criterion": Literal["gini", "entropy"],
        "max_depth": Optional[int],
        "min_samples_split": int,
        "min_samples_leaf": int,
        "max_features": Literal["sqrt", "log2", None],
        "bootstrap": bool,
        "oob_score": bool,
        "n_jobs": Optional[int],
        "random_state": Optional[int],
        "verbose": int,
        "warm_start": bool,
        "class_weight": Optional[dict],
        "ccp_alpha": float,
        "max_samples": Optional[int],
    },
)


@dataclass
class Prediction:
    X_pred: pd.DataFrame
    y_pred: pd.Series
    _y_true: Optional[pd.Series] = None

    def __post_init__(self):
        self.has_true = self._y_true

    @property
    def y_true(self):
        return self._y_true

    @y_true.setter
    def y_true(self, value):
        self._y_true = value



RandomForestModelDict = {
    "classifier": RandomForestClassifier,
    "regressor": RandomForestRegressor,
}

RandomForestModel = Union[RandomForestClassifier, RandomForestRegressor]


RandomForestModelTypes = Literal["classifier", "regressor"]


class PredictionsMissing(Exception):
    "Raise when prediction-based methods are called before making a prediction."

    def __init__(
        self,
        model: RandomForestModel,
        message: str = "No prediction has been made yet on this task. \
                          Please make a prediction first.",
    ) -> None:
        self.model = model
        self.message = message
        super().__init__()


class RandomForestModelTask:


    def __init__(self, model_type: RandomForestModelTypes = "classifier"):
        self.model_class = RandomForestModelDict[model_type]
        print(self.model_class)
        self.predictions: dict[Optional[int], Prediction] = {}

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

    @property
    def prediction_count(self):
        return len(self.predictions.keys())

    def get_prediction(self, prediction_id: int) -> Prediction:
        return self.predictions[prediction_id]

    @property
    def last_prediction(self):
        if self.predictions:
            return list(self.predictions.values())[-1]
        self._handle_missing_predictions()

    def feed_y_true(
        self, y_true: pd.Series, prediction_id: Optional[int] = None
    ) -> Prediction:
        if not self.predictions:
            self._handle_missing_predictions()
        prediction = self.predictions[prediction_id]
        if len(y_true) != len(prediction.y_pred):
            raise ValueError("Length of y_true must be equal to the length of y_pred.")
        prediction.y_true = y_true
        return prediction

    def create_model(self, model_params: Optional[RandomForestModelParams] = None) -> Self:
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

    def predict(
        self, X: Optional[pd.DataFrame] = None, y_true: Optional[pd.Series] = None
    ) -> Prediction:
        if not self._has_model():
            self._handle_has_not_model()
        check_is_fitted(self.model)
        if y_true is not None and X is not None:
            if y_true.size != X.shape[0]:
                raise ValueError("Length of y_true must be equal to the length of X.")
        X_pred = self.X_test if X is None else X
        prediction = Prediction(
            X_pred=X_pred, y_pred=self.model.predict(X_pred), _y_true=y_true
        )
        self.predictions[self.prediction_count] = prediction
        return prediction

    def get_classification_report(self, prediction_id: Optional[int] = None) -> None:
        if not self.predictions:
            self._handle_missing_predictions()
        if prediction_id is None:
            candidate_pred = self.last_prediction
        else:
            candidate_pred = self.get_prediction(prediction_id)
        if not candidate_pred.has_true:
            raise ValueError("No true values have been fed to this prediction.")
        return classification_report(
            candidate_pred.y_true, candidate_pred.y_pred, output_dict=True
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
