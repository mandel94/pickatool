import pandas as pd
from typing import Any
from dataclasses import dataclass
from formulaic import Formula

class Model:
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

    
@dataclass
class RegressionModelParams:
    """"""
    model_name: str
    formula: Formula
    model_type: str = "ols"


class RegressionModel(Model):
    def __init__(self, target: pd.DataFrame, features: pd.DataFrame, model_name: str, model_object: Any) -> None:
        super().__init__(target, features, model_name, model_object)

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
