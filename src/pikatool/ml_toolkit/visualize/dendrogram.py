import numpy as np
from scipy.cluster.hierarchy import dendrogram
from typing import Any, TypeAlias
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage


DistanceMatrix: TypeAlias = pd.DataFrame


def plot_dendrogram(
    dist_matrix: DistanceMatrix,
    method: str = "single",
    title: str = "Dendrogram plot",
    ax: Any = None,
    **dendrogramKw
) -> None:
    # fig, ax = plt.subplots(figsize=(23, 10))
    if ax is None:
        fig, ax = plt.subplots(figsize=(23, 10))
    if dendrogramKw is None:
        dendrogramKw = {}
    labels = dist_matrix.index
    mat = dist_matrix.to_numpy()
    dists = squareform(mat)  # Get condensed distance matrix
    linkage_matrix = linkage(dists, method)
    dendrogram(linkage_matrix, **dendrogramKw, labels=labels, ax=ax)
    ax.set_ylabel("Distance (min:0, max:1)", fontsize=20)
    # Remove figure frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # Add margin to y label title
    ax.yaxis.labelpad = 20
    # Change y ticks size
    ax.tick_params(axis="y", labelsize=15)
    # Add arrow at the end of y axis
    # Tight layout
    # plt.tight_layout()
    # plt.title(title, fontsize=25)
    # plt.show()
