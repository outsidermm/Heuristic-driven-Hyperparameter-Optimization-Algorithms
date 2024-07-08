import numpy as np
import math
import sys
sys.path.append(".")
from utility.DataAnalysis import min_max_scalar, weighted_avg

# Given data
epochss = np.array([20, 21, 22, 23, 24])
times = np.array([619, 3293, 5910, 8620, 11844])
accuracys = np.array([0.1596, 0.4106, 0.3854, 0.4106, 0.4256])


class EpochTuner:

    __epoch_list = np.array([])
    __time_list = np.array([])
    __accuracy_list = np.array([])

    __accuracy_normalized = [0]
    __time_normalized = [0]

    def __init__(
        self, left_bound: int, right_bound: int, exploration_factor: int = 1
    ) -> None:
        self.__left_bound = left_bound
        self.__right_bound = right_bound
        self.__exploration_factor = exploration_factor

    def epoch_run(self, epoch):
        # run the epoch
        acc = accuracys[epochss == epoch]
        time = times[epochss == epoch]

        self.__epoch_list = np.append(self.__epoch_list, epoch)
        self.__time_list = np.append(self.__time_list, time)
        self.__accuracy_list = np.append(self.__accuracy_list, acc)

        sorter = np.argsort(self.__epoch_list)
        self.__accuracy_list = self.__accuracy_list[sorter]
        self.__time_list = self.__time_list[sorter]
        self.__epoch_list = self.__epoch_list[sorter]

        if len(self.__epoch_list) > 1:
            self.__time_normalized = min_max_scalar(self.__time_list)
            self.__accuracy_normalized = min_max_scalar(self.__accuracy_list)

    def binary_search_efficient_epoch(self):
        left = self.__left_bound
        right = self.__right_bound
        best_epoch = self.__left_bound
        EpochTuner.epoch_run(self,self.__left_bound)
        EpochTuner.epoch_run(self,self.__right_bound)
        left_efficiency = weighted_avg(
            self.__time_normalized[self.__epoch_list == left][0],
            1 - self.__accuracy_normalized[self.__epoch_list == left][0],
        )
        right_efficiency = weighted_avg(
            self.__time_normalized[self.__epoch_list == right][0],
            1 - self.__accuracy_normalized[self.__epoch_list == right][0],
        )
        mid_efficiency = math.inf
        best_efficiency = math.inf

        while left <= right:
            # If the interval between left and right is smaller than the exploration factor, the lowest efficiency is found in the intervals
            if right - left <= self.__exploration_factor:
                if left_efficiency < best_efficiency:
                    best_efficiency = left_efficiency
                    best_epoch = self.__epoch_list[self.__epoch_list == left]
                if right_efficiency < best_efficiency:
                    best_efficiency = right_efficiency
                    best_epoch = self.__epoch_list[self.__epoch_list == right]
                if mid_efficiency < best_efficiency:
                    best_efficiency = mid_efficiency
                    best_epoch = self.__epoch_list[self.__epoch_list == mid]
                break

            mid = (left + right) // 2

            # Explore left, right and mid accuracy / time weighted average

            EpochTuner.epoch_run(self,mid)
            mid_efficiency = weighted_avg(
                self.__time_normalized[self.__epoch_list == mid][0],
                1 - self.__accuracy_normalized[self.__epoch_list == mid][0],
            )

            EpochTuner.epoch_run(self,left)
            left_efficiency = weighted_avg(
                self.__time_normalized[self.__epoch_list == left][0],
                1 - self.__accuracy_normalized[self.__epoch_list == left][0],
            )

            EpochTuner.epoch_run(self,right)
            right_efficiency = weighted_avg(
                self.__time_normalized[self.__epoch_list == right][0],
                1 - self.__accuracy_normalized[self.__epoch_list == right][0],
            )

            print(left_efficiency, mid_efficiency, right_efficiency)
            if left_efficiency < mid_efficiency:
                right = mid
            elif right_efficiency < mid_efficiency:
                left = mid
            # Heuristics: Typically the smaller the epoch, the more cost-effective it is, as time increases linearly while accuracy increases logarithmically
            else:
                right = mid

        return best_epoch, best_efficiency


epoch_tuner = EpochTuner(20, 24)
best_epoch, best_efficiency = epoch_tuner.binary_search_efficient_epoch()
print(best_epoch, best_efficiency)
