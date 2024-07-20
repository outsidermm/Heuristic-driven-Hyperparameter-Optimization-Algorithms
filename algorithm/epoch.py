import numpy as np
import math
import sys

sys.path.append(".")

from utility.helper_func import min_max_scalar, weighted_avg
from searcher import Searcher
from utility.dataloader import DataLoader
from typing import Tuple


class EpochTuner:

    __epoch_list = np.array([])
    __time_list = np.array([])
    __accuracy_list = np.array([])

    __accuracy_normalized = [0]
    __time_normalized = [0]

    def __init__(
        self,
        dataset: str,
        left_bound: int,
        right_bound: int,
        exploration_factor: int = 1,
    ) -> None:
        self.__left_bound = left_bound
        self.__right_bound = right_bound
        self.__exploration_factor = exploration_factor
        self.__dataset = dataset
        dataset_loader = DataLoader(self.__dataset)
        self.__train_ds, self.__val_ds, self.__test_ds = dataset_loader.load_dataset()

    def epoch_run(self, epoch: int) -> None:
        # run the epoch
        if epoch not in self.__epoch_list:
            time, acc = Searcher(
                dataset=self.__dataset,
                train_ds=self.__train_ds,
                val_ds=self.__val_ds,
                test_ds=self.__test_ds,
                verbose=1,
            ).training(epoch=epoch)

            self.__time_list = np.append(self.__time_list, time)
            self.__accuracy_list = np.append(self.__accuracy_list, acc)
            self.__epoch_list = np.append(self.__epoch_list, epoch)
            sorter = np.argsort(self.__epoch_list)
            self.__accuracy_list = self.__accuracy_list[sorter]
            self.__time_list = self.__time_list[sorter]
            self.__epoch_list = self.__epoch_list[sorter]

            if len(self.__epoch_list) > 1:
                self.__time_normalized = min_max_scalar(self.__time_list)
                self.__accuracy_normalized = min_max_scalar(self.__accuracy_list)

    def binary_search_efficient_epoch(self) -> Tuple[int, float, float]:
        left = self.__left_bound
        right = self.__right_bound
        best_epoch = self.__left_bound
        EpochTuner.epoch_run(self, self.__left_bound)
        EpochTuner.epoch_run(self, self.__right_bound)

        left_efficiency = self.weighted_avg_from_epoch(left)
        right_efficiency = self.weighted_avg_from_epoch(right)

        mid_efficiency = best_efficiency = math.inf

        while left <= right:
            # If the interval between left and right is smaller than the exploration factor, the lowest efficiency is found in the intervals
            if right - left <= self.__exploration_factor:
                for efficiency, epoch in [
                    (left_efficiency, left),
                    (right_efficiency, right),
                    (mid_efficiency, mid),
                ]:
                    if efficiency < best_efficiency:
                        best_efficiency = efficiency
                        best_epoch = self.__epoch_list[self.__epoch_list == epoch][0]
                break

            mid = (left + right) // 2

            # Explore left, right and mid accuracy / time weighted average

            EpochTuner.epoch_run(self, mid)
            EpochTuner.epoch_run(self, left)
            EpochTuner.epoch_run(self, right)

            mid_efficiency = self.weighted_avg_from_epoch(mid)
            left_efficiency = self.weighted_avg_from_epoch(left)
            right_efficiency = self.weighted_avg_from_epoch(right)

            if left_efficiency < mid_efficiency:
                right = mid
            elif right_efficiency < mid_efficiency:
                left = mid
            # Heuristics: Typically the smaller the epoch, the more cost-effective it is, as time increases linearly while accuracy increases logarithmically
            else:
                right = mid

        return (
            best_epoch[0],
            self.__accuracy_list[self.__epoch_list == best_epoch][0],
            self.__time_list[self.__epoch_list == best_epoch][0],
        )

    def weighted_avg_from_epoch(self, epoch: int) -> float:
        return weighted_avg(
            self.__time_normalized[self.__epoch_list == epoch][0],
            1 - self.__accuracy_normalized[self.__epoch_list == epoch][0],
        )
