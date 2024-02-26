import pandas as pd
from typing import Any, Optional, Literal, Union, Callable, TypeAlias
from .feature_clustering_implementations import FeatClustImplementations
from pickatool.stats_toolkit.compute_correlation import get_correlation_matrix_function

Array = Any

Distance: TypeAlias = Literal[
    "euclidean", "manhattan", "jaccard", "hamming", "matching"
]

CorrMatrixCallable: dict[Distance, Callable[[Array, Array], float]] = {
    dist: get_correlation_matrix_function(dist) for dist in Distance.__dict__["__args__"]
}

CorrelationMethod: TypeAlias = Union[Literal["pearson", "kendall", "spearman"], Distance]

GetClusterMethods = {
    "V0": FeatClustImplementations["V0"],
}

Cluster = Any


class CollinearityHandler:
    """"""

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.features = data.columns

    @staticmethod
    def compute_correlations(
        data: pd.DataFrame, method: CorrelationMethod = "pearson"
    ) -> pd.DataFrame:
        """Computes the correlation matrix of the features in the data.

        Args:
            method (CorrelationMethod, optional): The method for computing the
                correlation. Defaults to "pearson".

        Returns:
            pd.DataFrame: The correlation matrix of the features in the data.
        """
        if method in Distance.__dict__["__args__"]:
            return CorrMatrixCallable[method](data)   
        return data.corr(method=method)

    def get_clusters(self, threshold: float) -> list[Cluster]:
        """Computes the clusters of features based on a correlation matrix.

        Args:
            threshold (float): The threshold for the correlation.

        Returns:
            Any: The clusters of features based on the correlation matrix.
        """
        cor = self.cor
        clusters = FeatClustImplementations["V0"](cor, threshold)
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
        self.cor = CollinearityHandler.compute_correlations(
            self.data, method=compute_corr_with
        )

        self.clusters = self.get_clusters(threshold)

        return self.clusters
