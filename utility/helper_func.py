import csv
import numpy as np


def write_csv(data: list[dict], header: list[str], filename: str) -> None:
    with open(filename, mode="a+") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writerows(data)


def write_header(header: list[str], filename: str) -> None:
    with open(filename, mode="a+") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()


def min_max_scalar(data: np.array) -> np.array:
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def weighted_avg(
    val1: float, val2: float, weight1: float = 0.5, weight2: float = 0.5
) -> float:
    return val1 * weight1 + val2 * weight2
