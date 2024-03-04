from pickatool.stats_toolkit import MCA_v2 as MCA
from pickatool.eda_toolkit import DataHandler
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
        self.clusters: dict[ClusterName, Cluster] = (
            self._validate_clusters(clusters) if clusters else {}  # type: ignore
        )
        self.mca_results: dict[ClusterName, MCA_Result] = {}

    def _validate_clusters(
        self, clusters: OneOrMoreClusters
    ) -> dict[ClusterName, Cluster]:
        for cluster in clusters:
            for feature in cluster.nodes:
                if feature not in self.data.columns:
                    raise ValueError(f"Feature `{feature}` is not present in the data.")
        return {cluster.name: cluster for cluster in clusters}

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

