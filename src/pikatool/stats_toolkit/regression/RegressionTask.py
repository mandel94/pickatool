from dataclasses import dataclass
from typing import Union, Iterable, Any, TypeAlias, Optional, Self
import statsmodels.api as sm
from formulaic import Formula # https://pypi.org/project/formulaic/
import pandas as pd
import numpy as np


@dataclass
class RegressionModelParams:
    """"""
    model_name: str
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
    def __init__(self, target: pd.DataFrame, features: pd.DataFrame, model_name: str, model_object: Any) -> None:
        """Initializes the class with the target and features.
        
        Args:
            target (pd.DataFrame): The target variable for the regression model
            features (pd.DataFrame): The features to be used for the regression model
            model_object (Any): The underlying model object, like sm.OLS, sm.GLM, etc.
            model_name (str): The name of the model
            model_result (Any): The result of the model fit. 
            fitted (bool): Whether the model has been fitted or not
        """
        self.target = target
        self.features = features
        self.model_name = None
        self.model_object = model_object
        self.model_result = None
        self.fitted = False
    
    def fit(self, **kwargs) -> None:
        """Fits the model to the data.
        
        Returns the result object of the model fit.
        """
        res = self.model_object.fit(**kwargs)
        self.fitted = True
        self.model_result = res
    
    def summary(self) -> None:
        """Prints the summary of the model.
        """
        print(self.model_result.summary())





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


class RegressionTask():
    """"""

    def __init__(self, data: pd.DataFrame, task_name: Optional[str]="") -> None:
        """Initializes the class with the data.

        It gets data, and uses across all models that'll be created using this
        task instance. 
        
        Args:
            data (pd.DataFrame): The data to be used for the regression task
        """
        self.data = data
        self.models: dict[str, RegressionModel] = {}
        self.task_name = task_name


        
    def create_model(self, params: RegressionModelParams) -> RegressionModel:
        """Take data, and creates a regression model following parameters.

        params:
            RegressionModelParams: The parameters for the regression model.

        It returns the model created, and also stores it in the class instance
        """
        _id = params.model_name
        y, X = Formula(params.formula).get_model_matrix(self.data)
        self.models[_id] = RegressionModel(
            target=y,
            features=X,
            model_name = params.model_name if hasattr(params, "model_name") else None,
            model_object=_get_model(params.model_type)(y, X)
        )
        return self.models[_id]
    
    def run(self, model_id: Optional[Union[Iterable[int], int]] = None) -> Self:
        """Runs the model with the given id. 
        If no id is given, it runs all models added to the task.

        It returns the result of the model run.
        """
        if model_id is None:
            [self.models[id].fit() for id in self.models.keys()] # Run all models
        elif isinstance(model_id, int):
            model_id = [model_id]
        else:
            [self.models[id].fit() for id in model_id]
        return self
    

    def summary(self, model_id: int) -> None:
        """Prints the summary of the model with the given id.
        """
        return self.models[model_id].summary()
    

    def test_for_linearity(self, method: str = "rainbow") -> None:
        """Tests for linearity using the given method.
        """
        # https://www.statsmodels.org/stable/gettingstarted.html
        pass