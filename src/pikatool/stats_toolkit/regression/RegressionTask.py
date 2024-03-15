from dataclasses import dataclass
from typing import Union, Iterable, Any, TypeAlias, Optional, Self
import statsmodels.api as sm
from formulaic import Formula # https://pypi.org/project/formulaic/
import pandas as pd
import numpy as np
from pikatool.types import Model, RegressionModelParams, RegressionModel, RegressionResult



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
