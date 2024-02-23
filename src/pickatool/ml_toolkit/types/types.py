from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Union, Literal, Optional, TypedDict, Any, TypeAlias
from collections.abc import Mapping, Iterable


FineTuningParams: TypeAlias = Iterable[Mapping[str, Iterable[Any]]]
SearchStrategyName: TypeAlias = Literal[
    "grid", "random", "bayesian", "optuna", "hyperopt", "ray_tune"
]
SearchStrategy: TypeAlias = dict[str, Any]
RefitStrategy: TypeAlias = Literal["fast_precision_threshold_by_recall"]
TuningScoring: TypeAlias = Literal["accuracy", "precision", "recall", "f1"]
RandomForestModelTypes: TypeAlias = Literal["classifier", "regressor"]
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
RandomForestModel: TypeAlias = Union[RandomForestClassifier, RandomForestRegressor]
PredictiveModel: TypeAlias = Union[RandomForestModel]
