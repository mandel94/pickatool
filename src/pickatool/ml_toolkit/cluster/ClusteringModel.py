from dataclasses import dataclass
from .utils.plot_dendrogram import _plot_dendrogram
from typing import NamedTuple

from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.validation import check_is_fitted


ClusteringModelImpl = {"agglomerative": AgglomerativeClustering}


class ClusterTags(NamedTuple):
    clustered_entities: str
    cluster_tags: list[int]


@dataclass
class ClusteringModelParams:
    n_clusters: int
    linkage: {"ward", "complete", "average", "singles"} = "ward"
    metric: str = "euclidean"


class ClusteringModelTask:

    def __init__(self, model_type="agglomerative"):
        self.model_class = ClusteringModelImpl.get(model_type)

    # @staticmethod
    # def _get_model(model_class):
    #     return ClusteringModelImpl.get(model_class)

    @staticmethod
    def _attach_cluster_tags(
        data, cluster_tags: list[int], cluster_col_name: str = "Cluster"
    ):
        _data = data.copy(deep=True)
        _data[cluster_col_name] = cluster_tags
        return _data

    @property
    def enriched_data(self):
        # Chech if enriched data is available
        if not hasattr(self, "_enriched_data"):
            raise AttributeError(
                "Enriched data is not available. Please fit the model first."
            )
        return self._enriched_data

    def create_model(self, model_params: ClusteringModelParams):
        # TODO Validate type-specific parameteres
        self.model = self.model_class(**model_params, compute_distances=True)
        return self

    def fit_attach(self, data):
        self.model.fit(data)
        self._enriched_data = self._attach_cluster_tags(data, self.model.labels_)
        self.cluster_tags = ClusterTags(data.index, self.model.labels_)
        return ClusterTags(data.index, self.model.labels_)

    def plot_dendrogram(
        self,
        figure_parameters: dict = {},
        axes_parameters: dict = {},
        display_entities: bool = True,
        truncating_line: float = 0.7,
        **kwargs
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
            **kwargs
        )
