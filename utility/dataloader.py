from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader:
    """Loads pre-saved NumPy arrays from disk and returns TF datasets.

    Expects four ``.npy`` files under ``{dataset_dir}/{dataset}/``:
    ``X_train.npy``, ``y_train.npy``, ``X_test.npy``, ``y_test.npy``.

    Parameters
    ----------
    dataset : str
        Dataset name, e.g. ``"cifar100"`` or ``"imagenet"``.
    dataset_dir : str
        Root directory containing dataset subdirectories (default ``"./dataset"``).
    """

    def __init__(self, dataset: str, dataset_dir: str = "./dataset") -> None:
        self.__dataset = dataset
        self.__dataset_dir = dataset_dir

    def load_dataset(self):
        """Load and preprocess the dataset.

        Splits the original training set 60/40 into train/validation, applies
        pixel rescaling via the normalisation layer, and duplicates the training
        data once to augment the effective dataset size.

        Returns
        -------
        tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]
            Unbatched ``(train_ds, val_ds, test_ds)`` datasets.
        """
        import tensorflow as tf  # lazy import — TF is an optional dependency

        from utility.model import normalisation  # noqa: PLC0415

        base = f"{self.__dataset_dir}/{self.__dataset}"
        X_train: np.ndarray = np.load(f"{base}/X_train.npy")
        y_train: np.ndarray = np.load(f"{base}/y_train.npy")
        X_test: np.ndarray = np.load(f"{base}/X_test.npy")
        y_test: np.ndarray = np.load(f"{base}/y_test.npy")

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
