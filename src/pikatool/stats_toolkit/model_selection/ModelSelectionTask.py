from typing import Any, TypeAlias, TypeVar, Union, Iterable, Optional
from pikatool.types import Model


def _best_subset_selection(model: Model):
    """Best subset selection for linear regression."""
    target = model.target
    features = model.features
    n = len(features)
    # List of all possible models
    models = []
    best_models = Model(
        target=target,
        features=None,
        model_name="Null Model",
        model_object=model.model_object,
    )
    for i in range(1, n + 1):
        models.append(_get_best_k_subset(i, features, target))
    return models


FeatureList: TypeAlias = tuple[str, ...]


class ModelSelectionTask:
    """"""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.features = model.features
        self.target = model.target

    def best_subset_selection(self) -> Any:
        """Best subset selection for linear regression.

        Start with the null model (no predictors), then fit all simple linear regressions (one predictor at a time).
        Then fit all two-predictor models, all three-predictor models, and so forth.
        For each model, keep track of the RSS. Then select the best model with the lowest RSS.

        """
        return _best_subset_selection(self.model)
