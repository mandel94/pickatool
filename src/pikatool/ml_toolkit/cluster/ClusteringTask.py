from dataclasses import dataclass
from ..utils.plot_functions import (
    _plot_dendrogram,
    plot_silhouette as _plot_silhouette,
)
from typing import NamedTuple, Any, Self

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
            **kwargs: Additional parameters to be passed to the clustering evaluation function.

        Returns:
            float: Value of the clustering evaluation metric.
        """
        self._validate_metric(metric_)
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
        cluster_tags = [entity.cluster_tag for entity in self.clusters]
        metric_value = ClusteringEvaluationMetrics[metric_](
            self.data, cluster_tags, **kwargs
        )
        return metric_value


@dataclass
class ClusteringModelParams:
    n_clusters: int = 3
    linkage: {"ward", "complete", "average", "singles"} = "ward"
    metric: str = "euclidean"


class ClusteringTask:

    def __init__(self, model_type="agglomerative"):
        self.model_class = ClusteringModelImpl.get(model_type)

    def create_model(self, model_params: ClusteringModelParams = None):
        model_params = (
            model_params if model_params else ClusteringModelParams().__dict__
        )
        self.model = self.model_class(**model_params, compute_distances=True)
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
        self.model.fit(data)
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

    def plot_silhouette(self):
        self._plot_silhouette(
            self.clustering_result.data, self.cluster_tags, metric="jaccard"
        )
