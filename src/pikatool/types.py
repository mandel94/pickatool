import pandas as pd
from typing import Any
from dataclasses import dataclass, field
from formulaic import Formula
from typing import Literal, TypeAlias, Self
import numpy as np
from abc import abstractmethod


class Model:
    def __init__(
        self,
        target: pd.DataFrame,
        features: pd.DataFrame,
        model_name: str,
        model_object: Any,
    ) -> None:
        """Initializes the class with the target and features.

        Args:
            target (pd.DataFrame): The target variable for the regression model
            features (pd.DataFrame): The features to be used for the regression model
            model_object (Any): The underlying model object, like sm.OLS, sm.GLM, etc.
            model_name (str): The name of the model
            model_result (Any): The result of the model fit.
            fitted (bool): Whether the model has been fitted or not

        Methods:
            fit: Fits the model to the data
            get_coeffs_for_interpretation: Transforms the coefficients of the model into a more interpretable form.
            summary: Prints the summary of the model.
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

    @abstractmethod
    def get_coeffs_for_interpretation(self, family: Literal["regression"]) -> Self:
        """Transforms the coefficients of the model into a more interpretable form."""
        pass

    def summary(self) -> None:
        """Prints the summary of the model."""
        print(self.model_result.summary())


ModelTypes: TypeAlias = Literal["ols", "logit"]


class RegressionModelParams:
    """"""

    def __init__(
        self,
        model_name: str,
        model_type: str,
        predictors: list[str] = None,
        target: str = None,
        add_intercept: bool = True,
        formula: Formula = None,
    ) -> None:
        """Initializes the class with the parameters for the regression model."""
        self.model_name = model_name
        self.model_type = model_type
        self.predictors = predictors
        self.target = target
        self.add_intercept = add_intercept
        self.formula = formula


class RegressionModel(Model):
    """A class for regression models.

    It is a subclass of the Model class, and provides additional methods for
    interpreting the coefficients of the model.

    Args:
        target (pd.DataFrame): The target variable for the regression model
        features (pd.DataFrame): The features to be used for the regression model
        model_name (str): The name of the model
        model_object (Any): The underlying model object, like sm.OLS, sm.GLM, etc.
    
    Methods:
        get_coeffs_for_interpretation: Transforms the coefficients of the model into a more interpretable form.


    """
    def __init__(
        self,
        target: pd.DataFrame,
        features: pd.DataFrame,
        model_name: str,
        model_object: Any,
    ) -> None:
        super().__init__(target, features, model_name, model_object)

    def get_coeffs_for_interpretation(self) -> Any:
        """Transforms the coefficients of the model into a more interpretable form.s

        Returns the transformed coefficients. This is useful when I need to
        transform the coefficients for the sake of interpretation. For example,
        logit models have coefficients that are in log-odds: In that case it
        would be more useful to turn them into odd-ratios.


        Here is a list of the transformations that can be applied, for each model type:
        - ols: No transformation is needed
        - logit: The coefficients are transformed into increments in odds ratios
            log(p/(1-p)) = b0 + b1*x1 + b2*x2 + ... + bn*xn
            p/(1-p) = exp(b0 + b1*x1 + b2*x2 + ... + bn*xn)
            --> The coefficient can be interpreted as:
            ** numerical features: If you increase the feature by 1, the odds will increase by a factor of exp(coef)
            ** binary/categorical features: ... by a factor of exp(coef) compared to the reference category (0)
            ** intercept: When all features are 0, the odds will be exp(coef)
            https://christophm.github.io/interpretable-ml-book/logistic.html
        Example:
        ```python
        model.interpret_as("odds")
        ```
        """
        if not self.fitted:
            raise ValueError(
                "Model has not been fitted yet. Please run the fit method first."
            )
        if self.model_type == "logit":
            # return transformed coefficients
            return np.exp(self.model_result.params)
        else:
            raise ValueError("Model type not supported for better interpretation")


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
