import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import re


def _plot_dendrogram(
    model,
    entities: list = None,
    figure_parameters: dict = {},
    axes_parameters: dict = {},
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

    fig, ax = plt.subplots(**figure_parameters)
    
    xlabel_params = {
        key: re.search(r'(^xlabel_)(.*)', key)[1] for key in axes_parameters.keys()
    }
    ylabel_params = {
        key: re.search(r'(^ylabel_)(.*)', key)[1] for key in axes_parameters.keys()
    }
    ax.set_xlabel(**xlabel_params)
    ax.set_ylabel(**ylabel_params)
    # Remove parameters from the axes_parameters dict
    [axes_parameters.pop(key) for key in xlabel_params]
    [axes_parameters.pop(key) for key in ylabel_params]
    ax.set(**axes_parameters) # Remaining params
    if truncating_line:
        ax.axhline(y=truncating_line, c="r", linestyle="--")
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, 
               ax=ax, 
               labels=entities, 
               distance_sort="descending", 
               **kwargs)
