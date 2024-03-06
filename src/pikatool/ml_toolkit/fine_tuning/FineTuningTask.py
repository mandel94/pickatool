from ..types.types import (
    RefitStrategy,
)
from typing import Any, Callable, Optional, Self
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # type: ignore
from .custom_refits import fast_precision_threshold_by_recall

# https://scikit-learn.org/stable/modules/grid_search.html
# EXAMPLE OF NESTED CROSS VALIDATION: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html


class FineTuningTask:
    """Task for determining the best hyperparameters for a given model.

    Given a model, we need to feed it with training data. What the algorithm
    does is to split the training data into a training and validation set.
    Then, it trains the model with the training set and evaluates the model
    with the validation set, according to a given metric, such as accuracy,
    precision, recall, f1-score, etc.

    The algorithm will try different combinations of hyperparameters and will
    select the best combination of hyperparameters, according to the metric
    chosen.

    There are different algorithms for fine-tuning. Sklearn provides some of
    them, such as:
    - GridSearchCV
    - RandomizedSearchCV
    - BayesianOptimization

    Other libraries provide other algorithms, such as:
    - Optuna
    - Hyperopt
    - Ray Tune

    If you don't understand the difference between these algorithms, please
    refer to the documentation of each one of them.
    """

    SEARCH_OBJECTS = {
        "grid": GridSearchCV,
        "random": RandomizedSearchCV,
    }

    CUSTOM_REFITS = {
        "fast_precision_threshold_by_recall": fast_precision_threshold_by_recall
    }

    def __init__(self) -> None:
        pass

    def _handle_parameter_not_found(self, parameter: str) -> Any:
        """Condition: User feeds parameters that are not arguments of the
        fine-tuning task's model
        """
        pass

    def _get_params(self) -> Any:
        """Returns the parameters to fine-tune"""
        pass

    def _grid_search(self) -> Any:
        pass

    def _get_search_strategy(self, str) -> Any:
        pass

    def create_strategy(
        self, model, strategy_name, refit_strategy: Optional[RefitStrategy], **kwargs
    ) -> Self:
        """Define every aspect of the fine-tuning task. This returns a
        SearchStrategy object. Add it to the state of the class.

        Parameters
        ----------
        model : Any
            The model to fine-tune.
        strategy_name : SearchStrategyName
            The name of the strategy to use for fine-tuning.
        refit_strategy : Optional[RefitStrategy]
            The strategy to use for refitting the model after fine-tuning.
        **kwargs : All other parameters needed for the fine-tuning strategy
            accepted by the search object.

        Returns
        -------
        Self
            The object with the search strategy added to its state.

        """

        search_obj = FineTuningTask.SEARCH_OBJECTS[strategy_name]
        if refit_strategy and refit_strategy in RefitStrategy.__dict__["__args__"]:
            refit_fun = FineTuningTask.CUSTOM_REFITS[refit_strategy]
            self.search_strategy = search_obj(model, refit=refit_fun, **kwargs)
            return self
        self.search_strategy = search_obj(model, **kwargs)
        return self

    def run_strategy(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Run the fine-tuning strategy. Return an object with the search
        results, and add it to the state of the class.

        The strategy must have been previously defined by running the
        create_strategy method."""
        pass
