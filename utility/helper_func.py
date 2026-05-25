import csv

import numpy as np


def write_csv(data: list[dict], header: list[str], filename: str) -> None:
    """Append rows to a CSV file.

    Parameters
    ----------
    data : list[dict]
        Rows to write, each keyed by the column names in ``header``.
    header : list[str]
        Column names (must match keys in ``data``).
    filename : str
        Path to the target CSV file.
    """
    with open(filename, mode="a+") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writerows(data)


def write_header(header: list[str], filename: str) -> None:
    """Append a CSV header row to a file.

    Parameters
    ----------
    header : list[str]
        Column names to write as the header.
    filename : str
        Path to the target CSV file.
    """
    with open(filename, mode="a+") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()


def min_max_scalar(data: np.array) -> np.array:
    """Normalise an array to the [0, 1] range using min-max scaling.

    Parameters
    ----------
    data : np.array
        1-D array of numeric values.

    Returns
    -------
    np.array
        Scaled array with values in ``[0, 1]``.
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def weighted_avg(
    val1: float, val2: float, weight1: float = 0.5, weight2: float = 0.5
) -> float:
    """Compute a weighted average of two values.

    Parameters
    ----------
    val1, val2 : float
        Values to average.
    weight1, weight2 : float
        Weights for ``val1`` and ``val2`` respectively (default 0.5 each).

    Returns
    -------
    float
        ``val1 * weight1 + val2 * weight2``
    """
    return val1 * weight1 + val2 * weight2
