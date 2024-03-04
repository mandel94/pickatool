import pandas as pd
from typing import Any, Optional, Literal, Union, Callable, TypeAlias, NamedTuple
from .feature_clustering_implementations import FeatClustImplementations
from pikatool.stats_toolkit.compute_correlation import get_correlation_matrix_function
from ._types import FCluster as Cluster


Array = Any

Distance: TypeAlias = Literal[
    "euclidean", "manhattan", "jaccard", "hamming", "matching"
]

CorrMatrixCallable: dict[Distance, Callable[[Array, Array], float]] = {
    dist: get_correlation_matrix_function(dist)
    for dist in Distance.__dict__["__args__"]
}

CorrelationMethod: TypeAlias = Union[
    Literal["pearson", "kendall", "spearman"], Distance
]

GetClusterMethods = {
    "V0": FeatClustImplementations["V0"],
}


class CollinearityHandler:
    """"""

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.features = data.columns

    def compute_correlations(
        self, method: CorrelationMethod = "pearson"
    ) -> pd.DataFrame:
        """Computes the correlation matrix of the features in the data.

        Args:
            method (str, optional): The method for computing the correlation. Defaults to "pearson".

        Returns:
            pd.DataFrame: The correlation matrix of the features in the data.
        """
        if method in Distance.__dict__["__args__"]:
            self.corr = CorrMatrixCallable[method](self.data)
            return self.corr
        self.corr = self.data.corr(method=method)
        return self.corr

    def _compute_clusters(self, threshold: float = 0.7) -> list[Cluster]:
        """Computes the clusters of features based on a correlation matrix.

        Args:
            threshold (float, optional): The threshold for the correlation. Defaults to 0.7.

        Returns:
            Any: The clusters of features based on the correlation matrix.
        """
        clusters = FeatClustImplementations["V0"](self.corr, threshold)
        return clusters

    def cluster_features(
        self,
        threshold: Optional[float] = 0.7,
        compute_corr_with: CorrelationMethod = "pearson",
    ) -> Any:
        """Computes feature clusters based on a correlation method.

        The method for implementing clustering is based on the implementation
        of the `identify_cluster` function in the `collinearity_finder_treater`

        Args:
            threshold (float, optional): The threshold for the correlation.
                Defaults to 0.7.
            compute_corr_with (str, optional): The method for computing the
                correlation. Defaults to "pearson".

        Returns:

        """
        self.compute_correlations(method=compute_corr_with)
        self.clusters: list[Cluster] = []
        computed_clusters = self._compute_clusters(threshold)
        for cluster in computed_clusters:
            self.clusters.append(
                Cluster(name=cluster.name, nodes=cluster.nodes, pairs=cluster.pairs)
            )

        return self.clusters

    def print_clusters(self):
        """
        Print the clusters of features.
        """
        if not hasattr(self, "clusters"):
            raise ValueError(
                "No clusters have been computed. Call `cluster_features` first to compute clusters on features."
            )

        for cluster in self.clusters:
            print(
                f""" 
            Cluster: {cluster.name} 
            ---- Nodes: {cluster.nodes}
            ---- Pairs: {cluster.pairs}"""
            )
