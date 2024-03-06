from pikatool.stats_toolkit import MCA_v2 as MCA
from pikatool.eda_toolkit import DataHandler
from typing import Union, TypeAlias, Any, Literal, NamedTuple, Optional, Iterable, Self
import pandas as pd
from dataclasses import dataclass
from ._types import FCluster as Cluster

Data = Union[DataHandler, pd.DataFrame]


@dataclass
class MCA_Result:
    """The result of an MCA analysis."""

    eigenvalues: Any
    factored_individuals: Any
    factored_features: Any
    individuals_contributions: Any
    features_contributions: Any
    individuals_cosine_similarities: Any
    features_cosine_similarities: Any
    scree_plot: callable


ClusterName: TypeAlias = str
OneOrMoreClusters: TypeAlias = Union[Iterable[Cluster], Cluster]
MCA_ResultDict: dict[ClusterName, MCA_Result]


class mcaFC:
    """MCA-based Feature Clustering Task on a given dataset."""

    def __init__(
        self, data: Data, clusters: Optional[OneOrMoreClusters] = None
    ) -> None:
        if isinstance(data, DataHandler):
            tmp_df = data.data  # Need `tmp_df` to avoid mypy static analysis error
        tmp_df = data
        self.data = tmp_df
        del tmp_df
        self.clusters: dict[ClusterName, Cluster] = {}
        if clusters:
            for cluster in clusters:
                self.clusters[cluster.name] = self._validate_clusters(cluster)
        self.mca_results: dict[ClusterName, MCA_Result] = {}

    def _validate_clusters(self, cluster: Cluster) -> Cluster:
        # Which clusters have already been included in the MCA object?
        clustered_features = [
            feature for cluster in self.clusters.values() for feature in cluster.nodes
        ]
        for feature in cluster.nodes:
            # VALIDATION 1
            if feature in clustered_features:
                raise ValueError(
                    f"Feature `{feature}` is already present in some other cluster. Clusters shall not be overlapping"
                )
            # VALIDATION 2
            if feature not in self.data.columns:
                raise ValueError(f"Feature `{feature}` is not present in the data.")
        return cluster

    def feed_cluster(self, cluster: OneOrMoreClusters) -> Any:
        """Add clusters to the MCA object.

        Parameters
        ----------
        cluster : Union[Iterable[Cluster], Cluster]
            The cluster or clusters to add to the MCA object.
        """
        for c in cluster:
            self.clusters[c.name] = c

    def run(self, **kwargs) -> Self:
        """Run MCA analysis on all clusters.

        If a cluster has already been run, it will be skipped.

        Parameters
        ----------
        cluster : OneOrMoreClusters
            The cluster or clusters to run MCA on. It has been provided in the constructor, or
            it can be provided with the `self.feed_cluster` method`.

        Returns
        -------
        Self
            MCA object.
        """
        for c in self.clusters.values():
            # If cluster has already been run, skip it
            if c.name in self.mca_results.keys():
                continue
            res = MCA(self.data[list(c.nodes)], **kwargs)
            print(f"MCA on cluster `{c.name}` completed successfully.")
            self.mca_results[c.name] = MCA_Result(
                eigenvalues=res.eigenvalues,
                factored_individuals=res.row_coordinates,
                factored_features=res.column_coordinates,
                individuals_contributions=res.row_contributions.data,
                features_contributions=res.column_contributions.data,
                individuals_cosine_similarities=res.row_cosine_similarities,
                features_cosine_similarities=res.column_cosine_similarities,
                scree_plot=res.scree_plot_fun(c.name),
            )
        return self

    def plug_components(
        self, n_components: int = 1, inplace: bool = False
    ) -> pd.DataFrame:
        """Plug the MCA components into the current state of data.

        For each clusters that has been run, it substitutes the original features
        with the factored features.

        Args:
        inplace (bool, optional): If True, the original data will be modified.
            Defaults to False.

        Returns:
        pd.DataFrame: The modified data-state.
        """
        df_with_factors = self.data.copy(deep=True)
        for name, mca_res in self.mca_results.items():
            cluster = self.clusters[name]
            # Drop the original features
            df_with_factors.drop(columns=cluster.nodes, axis=1, inplace=True)
            for i in range(n_components):
                factored_individuals = mca_res.factored_individuals.iloc[:, i]
                df_with_factors[f"{name}_component_{i}"] = factored_individuals
        if inplace:
            self.data = df_with_factors
        return df_with_factors
