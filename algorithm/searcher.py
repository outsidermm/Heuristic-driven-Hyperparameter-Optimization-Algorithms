import sys

sys.path.append(".")

from keras import optimizers, losses, metrics, backend
from utility.model import VGG16
from utility.helper_func import write_csv, write_header
import time
import tensorflow as tf
from typing import Tuple

TOP_1 = metrics.TopKCategoricalAccuracy(k=1, name="Top_1")
TOP_5 = metrics.TopKCategoricalAccuracy(k=5, name="Top_5")
TOP_1_SPARSE = metrics.SparseTopKCategoricalAccuracy(k=1, name="Top_1")
TOP_5_SPARSE = metrics.SparseTopKCategoricalAccuracy(k=5, name="Top_5")
LOSS = losses.CategoricalCrossentropy()
LOSS_SPARSE = losses.SparseCategoricalCrossentropy()
AUTOTUNE = tf.data.experimental.AUTOTUNE


class Searcher:
    def __init__(
        self,
        dataset: str,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        verbose: int = 2,
    ) -> None:

        self.__dataset = dataset
        self.__distributed_strategy = tf.distribute.MirroredStrategy()
        self.__num_device = self.__distributed_strategy.num_replicas_in_sync
        self.__verbose = verbose
        self.__train_ds = train_ds
        self.__val_ds = val_ds
        self.__test_ds = test_ds
        self.__input_shape = (64, 64, 3) if dataset == "imagenet" else (32, 32, 3)
        self.__num_classes = 200 if self.__dataset == "imagenet" else 100
        backend.clear_session()

    def training(
        self,
        epoch: int = 250,
        batch_size: int = 128,
        lr: float = 0.01,
        momentum: float = 0.9,
    ) -> Tuple[float, float]:
        batch_size = batch_size * self.__num_device

        top_1_metrics = TOP_1 if self.__dataset == "imagenet" else TOP_1_SPARSE
        top_5_metrics = TOP_5 if self.__dataset == "imagenet" else TOP_5_SPARSE
        loss_function = LOSS if self.__dataset == "imagenet" else LOSS_SPARSE

        self.__train_ds = self.__train_ds.batch(batch_size)
        self.__val_ds = self.__val_ds.batch(batch_size)
        self.__test_ds = self.__test_ds.batch(batch_size)

        self.__train_ds = self.__train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.__val_ds = self.__val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.__test_ds = self.__test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        write_header(
            [
                "test_epoch",
                "test_batch_size",
                "test_lr",
                "test_momentum",
                "Time",
                "Accuracy",
                "Top1",
                "Top5",
            ],
            "./algorithm/" + self.__dataset + "/batch.csv",
        )

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

        metrics = cnn.evaluate(
            self.__test_ds,
            batch_size=batch_size,
            return_dict=True,
        )

        accuracy = metrics["accuracy"]
        top_1 = metrics["Top_1"]
        top_5 = metrics["Top_5"]

        print(f"Time taken: {time_taken}, Accuracy: {accuracy}")

        write_csv(
            [
                {
                    "test_epoch": epoch,
                    "test_batch_size": batch_size / self.__num_device,
                    "test_lr": lr,
                    "test_momentum": momentum,
                    "Time": time_taken,
                    "Accuracy": accuracy,
                    "Top1": top_1,
                    "Top5": top_5,
                }
            ],
            [
                "test_epoch",
                "test_batch_size",
                "test_lr",
                "test_momentum",
                "Time",
                "Accuracy",
                "Top1",
                "Top5",
            ],
            "./algorithm/" + self.__dataset + "/batch.csv",
        )

        return time_taken, accuracy
