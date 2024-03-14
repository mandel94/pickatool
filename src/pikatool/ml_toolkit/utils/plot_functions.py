import pandas as pd
import numpy as np
import re
from typing import Iterable
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import dendrogram


def plot_silhouette(
    data: pd.DataFrame,
    labels: Iterable[int | float],
    silhouette_score: float,
    metric: Optional[str] = "euclidean",
):
    # Create a subplot with 1 row and 2 columns
    fig, ax = plt.subplots()
    fig.set_size_inches(18, 7)

    n_clusters = len(set(labels))

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]

    # # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # # plots of individual clusters, to demarcate them clearly.
    # ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_score,
    )

    sample_silhouette_values = silhouette_samples(data, labels, metric=metric)

    x_upper_lim = max(sample_silhouette_values) + 0.1
    x_lower_lim = min(sample_silhouette_values) - 0.1
    ax.set_xlim(x_lower_lim, x_upper_lim)

    y_lower = 10

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        is_in_cluster = [l == i for l in labels]
        ith_cluster_silhouette_values = sample_silhouette_values[is_in_cluster]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_score, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yax1is labels / ticks

    plt.show()


def plot_dendrogram(
    model,
    entities: list = None,
    figure_parameters: dict = {},
    ax1es_parameters: dict = {},
    truncating_line: float = 0.7,
    **kwargs
):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    fig, ax1 = plt.subplots(**figure_parameters)

    xlabel_params = {
        key: re.search(r"(^xlabel_)(.*)", key)[1] for key in ax1es_parameters.keys()
    }
    ylabel_params = {
        key: re.search(r"(^ylabel_)(.*)", key)[1] for key in ax1es_parameters.keys()
    }
    ax1.set_xlabel(**xlabel_params)
    ax1.set_ylabel(**ylabel_params)
    # Remove parameters from the ax1es_parameters dict
    [ax1es_parameters.pop(key) for key in xlabel_params]
    [ax1es_parameters.pop(key) for key in ylabel_params]
    ax1.set(**ax1es_parameters)  # Remaining params
    if truncating_line:
        ax1.ax1hline(y=truncating_line, c="r", linestyle="--")
    # Plot the corresponding dendrogram
    dendrogram(
        linkage_matrix, ax1=ax1, labels=entities, distance_sort="descending", **kwargs
    )
    plt.show()


def plot_scores_grid(
    k_grid: Iterable[float | int], scoring_metric: Iterable[float | int]
):
    """Plot the scores of a clustering model against a grid of clusters' number.

    Args:
        k_grid (Iterable[float|int]): the grid of clusters' number.
        scoring_metric (Iterable[float|int]): the scores of the clustering model.
    """

    fig, ax = plt.subplots()
    # Plot a line. The x-axis is the number of clusters, the y-axis is the score.
    # Use a good looking theme, add labels and title. Add a void red circle in correspondence of the best score,
    # and a smaller cross for all other points.
    # Add an annotation to the best score point, with the annotation "Best score".

    ax.plot(k_grid, scoring_metric, marker="_", color="c")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Score")
    ax.set_title("Clustering score against number of clusters")
    best_score = max(scoring_metric)
    best_k = k_grid[scoring_metric.index(best_score)]
    ax.plot(best_k, best_score, marker="o", color="r")
    ax.annotate(
        "Best score",
        (best_k, best_score),
        textcoords="offset points",
        xytext=(-20, 10),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
        ha="center",
    )
    for k, score in zip(k_grid, scoring_metric):
        if k != best_k:
            ax.plot(k, score, marker="x", color="b")
    plt.show()
