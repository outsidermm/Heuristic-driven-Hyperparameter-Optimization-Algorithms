from collections.abc import Callable

import numpy as np

from utility.dataloader import DataLoader

from .searcher import Searcher


def _get_oom_error() -> type:
    """Return tf.errors.ResourceExhaustedError if TF is available, else a dummy."""
    try:
        import tensorflow as tf  # noqa: PLC0415

        return tf.errors.ResourceExhaustedError
    except ImportError:
        return type("_NeverRaised", (Exception,), {})


class BatchSizeTuner:
    """Heuristic binary-search tuner for the maximum viable batch size.

    Searches for the largest batch size (as a power of 2) whose accuracy stays
    within ``acceptable_range / 2`` of the baseline accuracy measured at
    ``left_bound``.

    Parameters
    ----------
    dataset : str
        Either ``"cifar100"`` or ``"imagenet"``.
    left_bound : int
        Log2 of the minimum batch size (e.g. ``4`` → batch size 16).
    right_bound : int
        Log2 of the maximum batch size (e.g. ``12`` → batch size 4096).
    acceptable_range : float
        Total acceptable accuracy drop relative to the baseline (default 0.30).
    training_fn : Callable[[int], tuple[float, float]] | None
        Optional override for the training function. Called as
        ``training_fn(batch_size)`` (actual batch size, not log2) and must
        return ``(time, accuracy)``. Pass a callable to run without TensorFlow.
    dataset_dir : str
        Root directory for dataset ``.npy`` files (default ``"./dataset"``).
        Ignored when ``training_fn`` is provided.
    """

    __batch_list = np.array([])
    __accuracy_list = np.array([])
    __time_list = np.array([])

    def __init__(
        self,
        dataset: str,
        left_bound: int,
        right_bound: int,
        acceptable_range: float = 0.30,
        training_fn: Callable[[int], tuple[float, float]] | None = None,
        dataset_dir: str = "./dataset",
    ) -> None:
        self.__acceptable_range = acceptable_range / 2
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

    def search(self) -> tuple[int, float, float]:
        """Search for the largest acceptable batch size.

        Returns
        -------
        Tuple[int, float, float]
            ``(best_batch_size, accuracy, time_taken_seconds)``.
            Returns ``(-1.0, -1.0, -1.0)`` if even the minimum batch size
            exceeds available GPU memory.
        """
        left = self.__left_bound
        _oom_error: type = _get_oom_error()
        try:
            _, acc_bound = self.batch_size_runner(left)
        except _oom_error:
            return -1.0, -1.0, -1.0

        right = self.__right_bound
        mid = left + (right - left) // 2

        while left <= right:
            try:
                time, acc = self.batch_size_runner(mid)

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

            except _oom_error:
                right = mid - 1

            mid = left + (right - left) // 2

        return (
            best_batch,
            self.__accuracy_list[self.__batch_list == best_batch][0],
            self.__time_list[self.__batch_list == best_batch][0],
        )

    def batch_size_runner(self, batch_size: int) -> tuple[float, float]:
        actual = 2**batch_size
        if self.__training_fn is not None:
            return self.__training_fn(actual)
        return Searcher(
            dataset=self.__dataset,
            train_ds=self.__train_ds,
            val_ds=self.__val_ds,
            test_ds=self.__test_ds,
            verbose=1,
        ).training(batch_size=actual)
