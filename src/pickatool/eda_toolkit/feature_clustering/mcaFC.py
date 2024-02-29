from pickatool.stats_toolkit import MCA
from pickatool.eda_toolkit import PandasDataHandler
from typing import Union, TypeAlias, Any, Literal, NamedTuple, Optional, Iterable, Self
import pandas as pd
from dataclasses import dataclass
from functools import lru_cache
from .types import FCluster as Cluster

Data = Union[PandasDataHandler, pd.DataFrame]



@dataclass
class MCA_Result:
    """The result of an MCA analysis.
    
    Attributes
    ----------
    eigenvalues : Union[pd.DataFrame|pd.Series]
        The eigenvalues of the MCA analysis. Each columns of the dataframe is an eigenvalue.
    """
    MCA_Object: MCA

ClusterName: TypeAlias = str
OneOrMoreClusters: TypeAlias = Union[Iterable[Cluster], Cluster]
MCA_ResultDict: dict[ClusterName, MCA_Result] 



class mcaFC():
    """Feature Clustering Task on a given dataset."""

    def __init__(self, data: Data, clusters: Optional[OneOrMoreClusters] = None) -> None:
        if isinstance(data, PandasDataHandler):
            data = data.data
        self.data = data
        self._validate_clusters(clusters)
        self.clusters: dict[str, Cluster] = clusters
        self.mca_results = {}
        self.processed_clusters: dict[ClusterName, Cluster] = {}

    def _validate_clusters(self, clusters: OneOrMoreClusters) -> None:
        """Clustered features must be present in `self.data.`"""
        if clusters is None:
            return
        for node in clusters.nodes:
            if node not in self.data.columns:
                raise ValueError(f"Feature {node} is not present in the data.")

    def feed_cluster(self, cluster: OneOrMoreClusters) -> list[Cluster]:
        """Add clusters to the MCA object.

        Parameters
        ----------
        cluster : Union[Iterable[Cluster], Cluster]
            The cluster or clusters to add to the MCA object.
        """
        if isinstance(cluster, Iterable):
            for c in cluster:
                self.clusters[c.name] = c
        else:
            self.clusters[cluster.name] = cluster

    @lru_cache
    def run(self, cluster: OneOrMoreClusters) -> Self:
        """Run MCA analysis on a cluster.

        Parameters
        ----------
        cluster : Union[Iterable[Cluster], Cluster]
            The cluster or clusters to run MCA on.
        """
        mca_result = MCA(self.data[cluster.nodes])
        self.processed_clusters_name.append(cluster.name)
        self.mca_results[cluster.name] = MCA_Result(mca_result)
        return self

    @lru_cache
    def run_all(self) -> Self:
        """Run MCA analysis on all clusters."""
        for cluster in self.clusters:
            [self.run(cluster) if cluster.name not in self.processed_clusters.keys() else None]
        return self