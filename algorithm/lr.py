import numpy as np
import sys

sys.path.append(".")

from searcher import Searcher
from utility.dataloader import DataLoader
from typing import Tuple
import tensorflow as tf


class LrTuner:

    __lr_list = np.array([])
    __accuracy_list = np.array([])
    __time_list = np.array([])

    def __init__(
        self,
        dataset: str,
        left_bound: int,
        right_bound: int,
        local_extrema_allowance: float = 0.05,
    ) -> None:
        self.__left_bound = left_bound
        self.__right_bound = right_bound
        self.__dataset = dataset
        dataset_loader = DataLoader(self.__dataset)
        self.__train_ds, self.__val_ds, self.__test_ds = dataset_loader.load_dataset()
        self.__local_extrema_allowance = local_extrema_allowance

    def search(self) -> Tuple[int, float, float]:
        left = self.__left_bound
        right = self.__right_bound

        while left <= right:
            mid = left + (right - left) // 2
            
            mid_acc, mid_time = self.data_stat_search(lr_linear=mid)
            left_acc,left_time = self.data_stat_search(lr_linear=left)
            right_acc,right_time = self.data_stat_search(lr_linear=right)

            if (mid_acc > left_acc + self.__local_extrema_allowance) and (
                mid_acc > right_acc + self.__local_extrema_allowance
            ):
                return mid, mid_acc, mid_time

            if mid_acc > left_acc:
                left = mid + 1
            else:
                right = mid - 1

        return -1, -1, -1

    def data_stat_search(self, lr_linear: float) -> Tuple[float,float]:
        if lr_linear not in self.__lr_list:
            time, acc = Searcher(
                dataset=self.__dataset,
                train_ds=self.__train_ds,
                val_ds=self.__val_ds,
                test_ds=self.__test_ds,
                verbose=1,
            ).training(lr=10 ** (-lr_linear))
            self.__lr_list = np.append(self.__lr_list, lr_linear)
            self.__accuracy_list = np.append(self.__accuracy_list, acc)
            self.__time_list = np.append(self.__time_list, time)
        else:
            acc = self.__accuracy_list[self.__lr_list == lr_linear][0]
            time = self.__time_list[self.__lr_list == lr_linear][0]
        
        return acc, time

