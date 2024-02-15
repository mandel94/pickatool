from ..utils.types import DataFrame, SquareDataframe, Node, Edge
from typing import Literal
from sklearn.metrics import DistanceMetric
import pandas as pd
import numpy as np



class NetworkAnalysis:
    def __init__(self, data: SquareDataframe, graph=None):
        self.data = data
        self.graph = graph

    @staticmethod
    def compute_distances(
        data: DataFrame,
        axis: Literal["columns", "rows"] = "columns",
        metric: str = "euclidean",
    ) -> SquareDataframe:
        distance_metric = DistanceMetric.get_metric(metric)
        if axis == "columns":
            index = data.columns
            distances = distance_metric.pairwise(data.T)
            return pd.DataFrame(distances, index=index, columns=index)
        else:
            index = data.index
            distances = distance_metric.pairwise(data)
            return pd.DataFrame(distances, index=index, columns=index)
    
    @property
    def nodes(self) -> list[Node]:
        nodes_ids = np.arange(0, len(self.data.index))
        # Convert str.data.index values to string
        nodes_names = self.data.index.astype(str)
        return list(zip(nodes_ids, nodes_names))
    
    @property
    def edges(self) -> list[Edge]:
        #(source, target, type, weight)
        # For each row of the dataframe, create an edge against each other row
        edges = []
        for i in range(len(self.data)):
            for j in range(len(self.data.iloc[i, :])):
                if i != j:
                    edges.append((i, j, "undirected", self.data.iloc[i, j]))
        return edges



