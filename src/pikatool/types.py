import pandas as pd

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