from ..types.types import RandomForestModel


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


class PredictionNotFound(Exception):
    "Raise when a prediction is not found in the predictions dictionary."

    def __init__(self, prediction_name: str) -> None:
        self.message = f"Prediction with name {prediction_name} not found"
        super().__init__()
