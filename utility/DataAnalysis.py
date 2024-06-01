import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_csv(filename: str):
    with open(filename, mode="r") as file:
        data = csv.DictReader(file)
        return list(data)


def write_csv(data: list[dict], header: list[str], filename: str) -> None:
    with open(filename, mode="a+") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writerows(data)


def write_header(header: list[str], filename: str) -> None:
    with open(filename, mode="a+") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()


def plot(data: list[dict], x: str, y: str):
    df = pd.DataFrame(data)
    xdata = np.array(df[x].tolist(), dtype=float)
    ydata = np.array(df[y].tolist(), dtype=float)
    print(xdata, ydata)
    plt.plot(xdata, ydata)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_sample(X, y, index):
    plt.imshow(X[index])
    plt.xlabel(y[index])
    plt.show()
