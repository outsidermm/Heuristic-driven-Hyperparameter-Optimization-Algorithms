import sys

sys.path.append(".")

import numpy as np
from sklearn.model_selection import train_test_split
from utility.model import normalisation
import tensorflow as tf
from typing import Tuple


class DataLoader:
    def __init__(self, dataset: str) -> None:
        self.__dataset = dataset

    def load_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        X_train: np.ndarray = np.load("./dataset/" + self.__dataset + "/X_train.npy")
        y_train: np.ndarray = np.load("./dataset/" + self.__dataset + "/y_train.npy")
        X_test: np.ndarray = np.load("./dataset/" + self.__dataset + "/X_test.npy")
        y_test: np.ndarray = np.load("./dataset/" + self.__dataset + "/y_test.npy")

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.40,
            random_state=42,
        )

        X_train = normalisation()(X_train)
        X_val = normalisation()(X_val)
        X_test = normalisation()(X_test)

        X_train = np.concatenate([X_train, X_train], axis=0)
        y_train = np.concatenate([y_train, y_train], axis=0)

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        return train_ds, val_ds, test_ds
