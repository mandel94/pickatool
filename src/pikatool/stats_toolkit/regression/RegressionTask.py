from dataclasses import dataclass
from typing import Union, Iterable, Any, TypeAlias, Optional, Self
import statsmodels.api as sm
from formulaic import Formula  # https://pypi.org/project/formulaic/
import pandas as pd
import numpy as np


@dataclass
class RegressionModelParams:
    """"""

    formula: Formula
    model_type: str = "ols"


class RegressionResult:
    """"""

    def __init__(self, params: Any, rsquared: float) -> None:
        """Results of regression

        Args:
            params (Any): The parameters of the model
            rsquared (float): The r-squared value of the model
        """
        self.params = params
        self.rsquared = rsquared


class RegressionModel:
    def __init__(
        self, target: pd.DataFrame, features: pd.DataFrame, model_object: Any
    ) -> None:
        """Initializes the class with the target and features.

        Args:
            target (pd.DataFrame): The target variable for the regression model
            features (pd.DataFrame): The features to be used for the regression model
            model_object (Any): The underlying model object, like sm.OLS, sm.GLM, etc.
        """
        self.target = target
        self.features = features
        self.model_object = model_object

    def fit(self, **kwargs) -> RegressionResult:
        """Fits the model to the data.

        Returns the result object of the model fit.
        """
        self.model.fit(**kwargs)

        return self.model


RegressionModels: TypeAlias = dict[id, RegressionModel]


def _get_model(type) -> Any:
    """Returns the model based on the type given.

    Args:
        type (str): The type of model to be created

    Returns:
        Any: The model object to be used for the regression model
    """
    if type == "ols":
        return sm.OLS
    elif type == "glm":
        return sm.GLM
    else:
        raise ValueError("Invalid model type")


class RegressionTask:
    """"""

    def __init__(self, data: pd.DataFrame) -> None:
        """Initializes the class with the data.

        It gets data, and uses across all models that'll be created using this
        task instance.

        Args:
            data (pd.DataFrame): The data to be used for the regression task
        """
        self.data = data
        self.models: RegressionModels = {}

    def create_model(self, params: RegressionModelParams) -> RegressionModel:
        """Take data, and creates a regression model following parameters.

        params:
            RegressionModelParams: The parameters for the regression model.

        It returns the model created, and also stores it in the class instance
        """
        id = len(
            self.models.keys()
        )  # Get current length of models dict (how many models have been created during this task?)
        y, X = Formula(params.formula).get_model_matrix(self.data)
        self.models[id] = RegressionModel(
            target=y, features=X, model_object=_get_model(params.model_type)(y, X)
        )
        return self.models[id]

    def run(self, model_id: Optional[Union[Iterable[int], int]]) -> Self:
        """Runs the model with the given id.
        If no id is given, it runs all models added to the task.

        It returns the result of the model run.
        """
        if model_id is None:
            [self.models[id].fit() for id in self.models.keys()]  # Run all models
        elif isinstance(model_id, int):
            model_id = [model_id]
        else:
            [self.models[id].fit() for id in model_id]
        return self

    def summary(self, model_id: int) -> None:
        """Prints the summary of the model with the given id."""
        return self.models[model_id].summary()

    def test_for_linearity(self, method: str = "rainbow") -> None:
        """Tests for linearity using the given method."""
        # https://www.statsmodels.org/stable/gettingstarted.html
        pass
