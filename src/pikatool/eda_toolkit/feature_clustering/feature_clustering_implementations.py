from .third_party.collinearity_finder_treater.cluster import cluster
from typing import Any, Literal, Union
import pandas as pd


def collinearity_finder_treater(cor: pd.DataFrame, threshold: float) -> list[cluster]:
    # This implementation is based on the implementation of the
    # `identify_cluster` function in the `collinearity_finder_treater` package.
    # For more information, see "https://www.kaggle.com/code/manueldeluzi/fixing-multicollinearity-by-feature-clustering/edit"

    clusters: list[Union[cluster, Any]] = []
    for j, col in enumerate(cor.columns):
        for i, row in enumerate(cor.columns[0:j]):
            if abs(cor.iloc[i, j]) > threshold:
                current_pair = (col, row, cor.iloc[i, j])
                current_pair_added = False
                for _c in clusters:
                    if _c.can_accept(current_pair):
                        _c.update_with(current_pair)
                        current_pair_added = True
                if current_pair_added == False:
                    clusters.append(cluster(pairs=[current_pair]))
    final_clusters = []

    # It is possible to have clusters with shared nodes which which is not desirable. Here we merge the cluster that share nodes.
    for _cluster in clusters:
        added_to_final = False
        for final_c in final_clusters:
            if _cluster.nodes.intersection(final_c.nodes) != set():
                final_c.merge_with_cluster(_cluster)
                added_to_final = True
        if added_to_final == False:
            final_clusters.append(_cluster)
    for i, _cluster in enumerate(final_clusters):
        _cluster.name = f"cluster_{i}"
    return final_clusters


FeatClustImplementations = {"V0": collinearity_finder_treater}
