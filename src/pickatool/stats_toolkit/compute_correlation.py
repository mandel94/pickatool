import numpy as np
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from scipy import stats
from typing import Callable
import pandas as pd


def compute_similarity(array1, array2, distance):
    """
    Compute the similarity between two arrays.

    Args:
        array1 (np.ndarray): The first array.
        array2 (np.ndarray): The second array.
        distance (str): The distance metric.

    Returns:
        float: The similarity between the two arrays.
    """
    return get_similarity_function(distance)(array1, array2)


def compute_correlation(array1, array2, method="pearson"):
    """
    Compute the correlation between two arrays.

    Args:
        array1 (np.ndarray): The first array.
        array2 (np.ndarray): The second array.
        method (str, optional): The method for computing the correlation. Defaults to "pearson".

    Returns:
        float: The correlation between the two arrays.
    """
    try:
        correlation_function = GetCorrelation[method]
    except KeyError:
        raise ValueError(
            f"Invalid method: {method}."
            "Available methods are: {list(GetCorrelation.keys())}"
        )
    return correlation_function(array1, array2)


def get_similarity_function(distance) -> Callable:
    """
    Get the similarity function based on the distance.

    Args:
        distance (str): The distance metric.

    Returns:
        Callable: The similarity function.
    """
    dist = DistanceMetric.get_metric(distance)
    def compute_similarity(array1, array2):
        array1 = pd.DataFrame(array1)
        array2 = pd.DataFrame(array2)
        return 1 - dist.pairwise(array1.T, array2.T)[0][0]
    return compute_similarity


def get_correlation_function(method="pearson") -> Callable:
    """
    Get the correlation function based on the method.

    Args:
        method (str, optional): The method for computing the correlation. Defaults to "pearson".

    Returns:
        Callable: The correlation function.
    """
    try:
        correlation_function = GetCorrelation[method]
    except KeyError:
        raise ValueError(
            f"Invalid correlation method: {method}. "
            f"Available methods are: {list(GetCorrelation.keys())}"
        )
    return correlation_function


def compute_correlation_matrix(data: pd.DataFrame, method="pearson") -> pd.DataFrame:
    """
    Compute the correlation matrix of the features in the data.

    Args:
        data (pd.DataFrame): The data.
        method (str, optional): The method for computing the correlation. Defaults to "pearson".

    Returns:
        pd.DataFrame: The correlation matrix of the features in the data.
    """
    # Initialize distance matrix
    distance_matrix = np.zeros((len(data.columns), len(data.columns)))
    # for each row, for each column, compute the correlation
    for i, col1 in enumerate(data.columns):
        for j, col2 in enumerate(data.columns):
            distance_matrix[i, j] = compute_correlation(
                data[col1], data[col2], method=method
            )
    distance_matrix = pd.DataFrame(distance_matrix, index=data.columns, columns=data.columns)
    return distance_matrix

def get_correlation_matrix_function(method="pearson") -> Callable:
    """
    Get the correlation matrix function based on the method.

    Args:
        method (str, optional): The method for computing the correlation. Defaults to "pearson".

    Returns:
        Callable: The correlation matrix function.
    """ 
    return lambda data: compute_correlation_matrix(data, method=method)

GetCorrelation = {
    "pearson": lambda array1, array2: stats.pearsonr(array1, array2)[0],
    "kendall": lambda array1, array2: stats.kendalltau(array1, array2)[0],
    "spearman": lambda array1, array2: stats.spearmanr(array1, array2)[0],
    **{
        distance: get_similarity_function(distance)
        for distance in [
            "euclidean",
            "manhattan",
            "jaccard",
            "dice",
            "hamming",
            "matching",
            "kulsinski",
        ]
    },
}


        
