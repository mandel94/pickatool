from dataclasses import dataclass
from ..utils import (
    plot_dendrogram as _plot_dendrogram,
    plot_silhouette as _plot_silhouette,
    plot_scores_grid as _plot_scores_grid,
)
from typing import NamedTuple, Any, Self, Optional, Sequence, Tuple

from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import pandas as pd


ClusteringModelImpl = {"agglomerative": AgglomerativeClustering}

ClusteringEvaluationMetrics = {
    "silhouette_score": silhouette_score,
    "calinski_harabasz_score": calinski_harabasz_score,
    "davies_bouldin_score": davies_bouldin_score,
}


class ClusterTag(NamedTuple):
    entity_id: Any
    cluster_tag: int | float


@dataclass
class ClusteringResult:
    """Class to store the clustering result.

    If stores the data and the cluster tags.

    Each instance of this class is intended to store the result of a single clustering task.

    Args:
        data (pd.DataFrame): data to be clustered.
        clusters (list[ClusterTag]): list of cluster tags.

    Methods:
        __call__: Enriches the data with the cluster tags. Returns the enriched data.
        get_metric: Returns the value of a clustering evaluation metric.

    """

    data: pd.DataFrame
    clusters: list[ClusterTag]

    def __post_init__(self):
        self.validate_clustering_result()

    def clustering_score(self, metric_, **kwargs) -> float:
        """Evaluate the clustering result using a clustering evaluation metric.

        Args:
            metric_ (str): Name of the clustering evaluation metric to be used.
                Allowed values are:
                - "silhouette_score"
                - "calinski_harabasz_score"
                - "davies_bouldin_score"
                - A callable function that takes the data and the cluster tags as input and returns a float.

            **kwargs: Additional parameters to be passed to the clustering evaluation function.

        Returns:
            float: Value of the clustering evaluation metric.
        """
        self._validate_metric(metric_)
        if callable(metric_):
            return metric_(self.data, self.cluster_tags, **kwargs)
        return self._evaluate_clusters(metric_, **kwargs)

    def _validate_metric(self, metric_: str):
        if metric_ not in ClusteringEvaluationMetrics.keys():
            raise ValueError(
                f"Invalid metric: {metric_}. Please choose one of the following: {ClusteringEvaluationMetrics.keys()}"
            )

    def validate_clustering_result(self):
        # Check that all entities in the data are clustered
        entities_id = [c.entity_id for c in self.clusters]
        assert entities_id == list(
            self.data.index
        ), "Entities in clusters differ from entities in data."

    def __call__(self) -> pd.DataFrame:
        self.data["k"] = [c.cluster_tag for c in self.clusters]
        return self.data
    
    @property
    def cluster_tags(self) -> list[float | int]:
        "Returns labels of the clusters."
        return [c.cluster_tag for c in self.clusters]

    def _evaluate_clusters(self, metric_: str = "silhouette_score", **kwargs) -> float:
        """Evaluate the clustering result using a clustering evaluation metric.

        It takes `self.goodness_metrics` dictionary attribute, and adds the value of the
        clustering evaluation metric to it.

        Args:
            metric (str): Name of the clustering evaluation metric to be used.
                Allowed values are:
                - "silhouette_score"
                - "calinski_harabasz_score"
                - "davies_bouldin_score"

            **kwargs: Additional parameters to be passed to the clustering evaluation function.

        Returns:
            float: Value of the clustering evaluation metric.
        """
        if metric_ not in ClusteringEvaluationMetrics.keys():
            raise ValueError(
                f"Invalid metric: {metric_}. Please choose one of the following: {ClusteringEvaluationMetrics.keys()}"
            )
        metric_func = ClusteringEvaluationMetrics[metric_]
        # Only keep variables that are not present in the metric function signature
        kwargs = {k: v for k, v in kwargs.items() if k in metric_func.__code__.co_varnames}
        metric_value = ClusteringEvaluationMetrics[metric_](
            X=self.data,
            labels=self.cluster_tags, 
            **kwargs
        )
        return metric_value


@dataclass
class ClusteringModelParams:
    n_clusters: int = 3
    linkage: {"ward", "complete", "average", "singles"} = "ward"
    metric: str = "euclidean"


class ClusteringTask:

    def __init__(self, data: Optional[pd.DataFrame] = None, model_type: str ="agglomerative"):
        self.model_class = ClusteringModelImpl.get(model_type)
        self.data = data  
        self.has_model = False
        self.is_fitted = False
        self.data = None
        self.clustering_params = ClusteringModelParams()

    def create_model(self, model_params: ClusteringModelParams = None):
        self.model_params = model_params
        self.model = self.model_class(**self.model_params.__dict__, compute_distances=True)
        return self

    def fit_attach(self, data) -> Self:
        """Fit the model and attach cluster tags to the data.

        Takes data and fits the model to it. Then, it attaches the cluster tags to the data,
        so that the data is enriched with the cluster tags. The enriched data is
        then stored together with the cluster tags in a ClusteringResult instance.

        Each time this method is called, the previous clustering result is overwritten.
        If you want to test different clustering models or parameters, you should create
        new instances of ClusteringTask respectively.

        Args:
            data (pd.DataFrame): data to be clustered.

        Returns:
            Self: ClusteringTask instance.
        """
        if not data:
            raise ValueError("No data provided.")
        self.data = data
        self.model.fit(data)
        self.is_fitted = True
        self.clustered_tags = []
        for entity, cl_label in zip(data.index, self.model.labels_):
            self.clustered_tags.append(ClusterTag(entity, cl_label))
        self.clustering_result = ClusteringResult(data, self.clustered_tags)
        return self

    @property
    def cluster_tags(self) -> list[float | int]:
        "Returns labels of the clusters."
        return [c.cluster_tag for c in self.clustered_tags]

    def plot_dendrogram(
        self,
        figure_parameters: dict = {},
        axes_parameters: dict = {},
        display_entities: bool = True,
        truncating_line: float = 0.7,
        **kwargs,
    ):
        check_is_fitted(self.model)
        if display_entities:
            entities = self.cluster_tags.clustered_entities
        else:
            entities = None
        _plot_dendrogram(
            self.model,
            entities,
            figure_parameters,
            axes_parameters,
            truncating_line,
            **kwargs,
        )



class ClusteringScorer: 
    """"""
    
    def __init__(self, task: ClusteringTask):
        self.task = task

    def _check_has_results_to_plot(self):
        if not self.task.is_fitted:
            raise ValueError("The task has no fitted model. You must have some" 
                             "clustered data to plot.")


    def plot_silhouette(self, metric: Optional[str] = "euclidean"):
        """Plot silhouette plot for the clustering result.

        Args:
            metric (str): The metric to be used to calculate the silhouette score.
                The metric must be one of the metrics allowed by the `sklearn.metrics.silhouette_score` function.
                Defaults to "euclidean".

        """
        self._check_has_results_to_plot()

        tags = [t for t in self.task.clustered_tags]

        _plot_silhouette(
            self.task.clustering_result.data,
            self.task.cluster_tags,
            silhouette_score=self.task.clustering_result.clustering_score(
                "silhouette_score",
                metric=metric,
            ),
            metric=metric,
        )
    
   
    def plot_dendrogram(
        self,
        figure_parameters: dict = {},
        axes_parameters: dict = {},
        display_entities: bool = True,
        truncating_line: float = 0.7,
        **kwargs,
    ):
        self._check_has_results_to_plot()
        
        if display_entities:
            entities = self.task.cluster_tags.clustered_entities
        else:
            entities = None
        _plot_dendrogram(
            self.task.model,
            entities,
            figure_parameters,
            axes_parameters,
            truncating_line,
            **kwargs,
        )     


    def plot_scores_grid(self, scoring_metric: str = "silhouette_score",
                         range_: Sequence[Tuple[int, int]] = [3, 12], **kwargs) -> None:
        """Plot results of clustering against a predefined grid of parameters.
        
        This method works by using task to fit different models with different number of clusters.
        All other clustering parameters are kept as they are in the task.
        For each model, the scoring metric is calculated, and the list of 
        (number of clusters, scoring metric value) is used to create the plot.

        Args:
            scoring_metric (str): the scoring metric to plot the elbow plot for.
                The metric must be one of the metrics allowed by the `sklearn.metrics` module.
                Defaults to "silhouette_score".
            **kwargs: Additional parameters to be passed to the clustering evaluation function.
        """

        results = []
        for k in range(range_[0], range_[1]):
            self.task.create_model(ClusteringModelParams(n_clusters=k))
            self.task.fit_attach(self.task.data)
            import warnings
            with warnings.catch_warnings(action="ignore"):
                score = self.task.clustering_result.clustering_score(scoring_metric, **kwargs)
            results.append((k, score))
        k_grid = [r[0] for r in results]
        scores = [r[1] for r in results]
        _plot_scores_grid(k_grid, scores)



