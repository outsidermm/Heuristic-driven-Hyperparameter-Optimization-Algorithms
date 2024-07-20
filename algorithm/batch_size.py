import numpy as np
import sys

sys.path.append(".")

from searcher import Searcher
from utility.dataloader import DataLoader
from typing import Tuple
import tensorflow as tf


class BatchSizeTuner:

    __batch_list = np.array([])
    __accuracy_list = np.array([])
    __time_list = np.array([])

    def __init__(
        self,
        dataset: str,
        left_bound: int,
        right_bound: int,
        acceptable_range: float = 0.30,
    ) -> None:
        self.__acceptable_range = acceptable_range / 2
        self.__left_bound = left_bound
        self.__right_bound = right_bound
        self.__dataset = dataset
        dataset_loader = DataLoader(self.__dataset)
        self.__train_ds, self.__val_ds, self.__test_ds = dataset_loader.load_dataset()

    def search(self) -> Tuple[int, float, float]:
        left = self.__left_bound
        try:
            _, acc_bound = Searcher(
                dataset=self.__dataset,
                train_ds=self.__train_ds,
                val_ds=self.__val_ds,
                test_ds=self.__test_ds,
                verbose=1,
            ).training(batch_size=2**left)
        except tf.errors.ResourceExhaustedError:
            return -1.0, -1.0, -1.0

        right = self.__right_bound
        mid = left + (right - left) // 2

        while left <= right:
            try:
                time, acc = Searcher(
                    dataset=self.__dataset,
                    train_ds=self.__train_ds,
                    val_ds=self.__val_ds,
                    test_ds=self.__test_ds,
                    verbose=1,
                ).training(batch_size=2**mid)

                if mid not in self.__batch_list:
                    self.__time_list = np.append(self.__time_list, time)
                    self.__accuracy_list = np.append(self.__accuracy_list, acc)
                    self.__batch_list = np.append(self.__batch_list, mid)

                if abs(acc - acc_bound) < self.__acceptable_range:
                    acc_bound = (acc + acc_bound) / 2
                    best_batch = mid
                    left = mid + 1
                else:
                    right = mid - 1

            except tf.errors.ResourceExhaustedError:
                right = mid - 1

            mid = left + (right - left) // 2

        return (
            best_batch,
            self.__accuracy_list[self.__batch_list == best_batch][0],
            self.__time_list[self.__batch_list == best_batch][0],
        )
