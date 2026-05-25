import logging
import time
from pathlib import Path

from utility.helper_func import write_csv, write_header

logger = logging.getLogger(__name__)

CSV_HEADER = [
    "test_epoch", "test_batch_size", "test_lr", "test_momentum",
    "Time", "Accuracy", "Top1", "Top5",
]


class Searcher:
    """Single-run trainer for a VGG16 model on CIFAR-100 or ImageNet.

    Parameters
    ----------
    dataset : str
        Either ``"cifar100"`` or ``"imagenet"``.
    train_ds : tf.data.Dataset
        Unbatched training dataset of ``(image, label)`` pairs.
    val_ds : tf.data.Dataset
        Unbatched validation dataset.
    test_ds : tf.data.Dataset
        Unbatched test dataset.
    verbose : int
        Keras verbosity level passed to ``model.fit``.
    """

    def __init__(
        self,
        dataset: str,
        train_ds,
        val_ds,
        test_ds,
        verbose: int = 2,
    ) -> None:
        import tensorflow as tf  # lazy import — TF is an optional dependency
        from keras import backend

        self.__dataset = dataset
        self.__output_dir = Path("./algorithm") / dataset
        self.__distributed_strategy = tf.distribute.MirroredStrategy()
        self.__num_device = self.__distributed_strategy.num_replicas_in_sync
        self.__verbose = verbose
        self.__train_ds = train_ds
        self.__val_ds = val_ds
        self.__test_ds = test_ds
        self.__input_shape = (64, 64, 3) if dataset == "imagenet" else (32, 32, 3)
        self.__num_classes = 200 if self.__dataset == "imagenet" else 100
        self.__output_dir.mkdir(parents=True, exist_ok=True)
        backend.clear_session()

    def training(
        self,
        epoch: int = 250,
        batch_size: int = 128,
        lr: float = 0.01,
        momentum: float = 0.9,
    ) -> tuple[float, float]:
        """Train the model and return elapsed time and test accuracy.

        Parameters
        ----------
        epoch : int
            Number of training epochs.
        batch_size : int
            Per-device batch size (automatically scaled by the number of GPUs).
        lr : float
            SGD learning rate.
        momentum : float
            SGD momentum.

        Returns
        -------
        tuple[float, float]
            ``(time_taken_seconds, test_accuracy)``
        """
        import tensorflow as tf  # noqa: PLC0415
        from keras import backend, losses, metrics, optimizers  # noqa: PLC0415

        from utility.model import VGG16  # noqa: PLC0415

        top_1 = metrics.TopKCategoricalAccuracy(k=1, name="Top_1")
        top_5 = metrics.TopKCategoricalAccuracy(k=5, name="Top_5")
        top_1_sparse = metrics.SparseTopKCategoricalAccuracy(k=1, name="Top_1")
        top_5_sparse = metrics.SparseTopKCategoricalAccuracy(k=5, name="Top_5")

        autotune = tf.data.experimental.AUTOTUNE
        top_1_metrics = top_1 if self.__dataset == "imagenet" else top_1_sparse
        top_5_metrics = top_5 if self.__dataset == "imagenet" else top_5_sparse
        loss_function = (
            losses.CategoricalCrossentropy()
            if self.__dataset == "imagenet"
            else losses.SparseCategoricalCrossentropy()
        )

        batch_size = batch_size * self.__num_device

        self.__train_ds = self.__train_ds.batch(batch_size)
        self.__val_ds = self.__val_ds.batch(batch_size)
        self.__test_ds = self.__test_ds.batch(batch_size)

        self.__train_ds = self.__train_ds.cache().prefetch(buffer_size=autotune)
        self.__val_ds = self.__val_ds.cache().prefetch(buffer_size=autotune)
        self.__test_ds = self.__test_ds.cache().prefetch(buffer_size=autotune)

        write_header(CSV_HEADER, str(self.__output_dir / "lr.csv"))

        backend.clear_session()

        with self.__distributed_strategy.scope():
            cnn = VGG16(input_shape=self.__input_shape, num_class=self.__num_classes)
            cnn.summary()
            cnn.compile(
                optimizer=optimizers.SGD(learning_rate=lr, momentum=momentum),
                loss=loss_function,
                metrics=["accuracy", top_1_metrics, top_5_metrics],
            )

        start = time.time()
        cnn.fit(
            self.__train_ds,
            batch_size=batch_size,
            epochs=epoch,
            verbose=self.__verbose,
            validation_data=(self.__val_ds),
        )
        end = time.time()
        time_taken = end - start

        eval_metrics = cnn.evaluate(
            self.__test_ds,
            batch_size=batch_size,
            return_dict=True,
        )

        accuracy = eval_metrics["accuracy"]
        top_1_score = eval_metrics["Top_1"]
        top_5_score = eval_metrics["Top_5"]

        logger.info("Time taken: %.2f s, Accuracy: %.4f", time_taken, accuracy)

        write_csv(
            [
                {
                    "test_epoch": epoch,
                    "test_batch_size": batch_size / self.__num_device,
                    "test_lr": lr,
                    "test_momentum": momentum,
                    "Time": time_taken,
                    "Accuracy": accuracy,
                    "Top1": top_1_score,
                    "Top5": top_5_score,
                }
            ],
            CSV_HEADER,
            str(self.__output_dir / "lr.csv"),
        )

        return time_taken, accuracy
