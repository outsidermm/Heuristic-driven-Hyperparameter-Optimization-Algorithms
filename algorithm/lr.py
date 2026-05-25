from collections.abc import Callable

import numpy as np

from utility.dataloader import DataLoader

from .searcher import Searcher


class LrTuner:
    """Heuristic binary-search tuner for the optimal learning rate.

    Searches the log-linear space ``[10^-left_bound, 10^-right_bound]`` for a
    local accuracy maximum, stopping when a point exceeds both its neighbours
    by at least ``local_extrema_allowance``.

    Parameters
    ----------
    dataset : str
        Either ``"cifar100"`` or ``"imagenet"``.
    left_bound : int
        Negative exponent lower bound (e.g. ``1`` → lr starts at ``0.1``).
    right_bound : int
        Negative exponent upper bound (e.g. ``7`` → lr ends at ``1e-7``).
    local_extrema_allowance : float
        Minimum accuracy gap required to declare a local maximum (default 0.05).
    training_fn : Callable[[float], tuple[float, float]] | None
        Optional override for the training function. Called as
        ``training_fn(lr)`` (actual learning rate value) and must return
        ``(time, accuracy)``. Pass a callable to run without TensorFlow.
    dataset_dir : str
        Root directory for dataset ``.npy`` files (default ``"./dataset"``).
        Ignored when ``training_fn`` is provided.
    """

    __lr_list = np.array([])
    __accuracy_list = np.array([])
    __time_list = np.array([])

    def __init__(
        self,
        dataset: str,
        left_bound: int,
        right_bound: int,
        local_extrema_allowance: float = 0.05,
        training_fn: Callable[[float], tuple[float, float]] | None = None,
        dataset_dir: str = "./dataset",
    ) -> None:
        self.__left_bound = left_bound
        self.__right_bound = right_bound
        self.__dataset = dataset
        if training_fn is not None:
            self.__training_fn = training_fn
            self.__train_ds = self.__val_ds = self.__test_ds = None
        else:
            dataset_loader = DataLoader(self.__dataset, dataset_dir=dataset_dir)
            self.__train_ds, self.__val_ds, self.__test_ds = dataset_loader.load_dataset()
            self.__training_fn = None
        self.__local_extrema_allowance = local_extrema_allowance

    def search(self) -> tuple[int, float, float]:
        """Search for the learning rate with a local accuracy maximum.

        Returns
        -------
        Tuple[int, float, float]
            ``(lr_linear_exponent, accuracy, time_taken_seconds)``.
            Returns ``(-1, -1, -1)`` if no local maximum is found.
        """
        left = self.__left_bound
        right = self.__right_bound

        while left <= right:
            mid = left + (right - left) // 2

            mid_acc, mid_time = self.data_stat_search(lr_linear=mid)
            left_acc, left_time = self.data_stat_search(lr_linear=left)
            right_acc, right_time = self.data_stat_search(lr_linear=right)

            if (mid_acc > left_acc + self.__local_extrema_allowance) and (
                mid_acc > right_acc + self.__local_extrema_allowance
            ):
                return mid, mid_acc, mid_time

            if mid_acc > left_acc:
                left = mid + 1
            else:
                right = mid - 1

        return -1, -1, -1

    def data_stat_search(self, lr_linear: float) -> tuple[float, float]:
        if lr_linear not in self.__lr_list:
            lr = 10 ** (-lr_linear)
            if self.__training_fn is not None:
                time, acc = self.__training_fn(lr)
            else:
                time, acc = Searcher(
                    dataset=self.__dataset,
                    train_ds=self.__train_ds,
                    val_ds=self.__val_ds,
                    test_ds=self.__test_ds,
                    verbose=1,
                ).training(lr=lr)
            self.__lr_list = np.append(self.__lr_list, lr_linear)
            self.__accuracy_list = np.append(self.__accuracy_list, acc)
            self.__time_list = np.append(self.__time_list, time)
        else:
            acc = self.__accuracy_list[self.__lr_list == lr_linear][0]
            time = self.__time_list[self.__lr_list == lr_linear][0]

        return acc, time
