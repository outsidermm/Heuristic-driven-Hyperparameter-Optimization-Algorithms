import sys

sys.path.append(".")

from keras import optimizers, losses, metrics, backend
from utility.model import VGG16
from utility.helper_func import write_csv, write_header
import time
import tensorflow as tf

TOP_1 = metrics.TopKCategoricalAccuracy(k=1, name="Top_1")
TOP_5 = metrics.TopKCategoricalAccuracy(k=5, name="Top_5")
TOP_1_SPARSE = metrics.SparseTopKCategoricalAccuracy(k=1, name="Top_1")
TOP_5_SPARSE = metrics.SparseTopKCategoricalAccuracy(k=5, name="Top_5")
LOSS = losses.CategoricalCrossentropy()
LOSS_SPARSE = losses.SparseCategoricalCrossentropy()
AUTOTUNE = tf.data.experimental.AUTOTUNE


class GridSearch:

    def __init__(
        self,
        dataset: str,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        epoch_range: list[int] = [250],
        batch_size_range: list[int] = [
            16,
            64,
            256,
            1024,
            4096,
            16384,
            65536,
            262144,
            1048576,
            4194304,
            16777216,
        ],
        lr_range: list[float] = [0.01],
        momentum_range: list[float] = [0.9],
        verbose: int = 2,
    ) -> None:
        self.__input_shape = (64, 64, 3) if dataset == "imagenet" else (32, 32, 3)
        self.__dataset = dataset
        self.__num_classes = 200 if self.__dataset == "imagenet" else 100
        self.__distributed_strategy = tf.distribute.MirroredStrategy()
        self.__num_device = self.__distributed_strategy.num_replicas_in_sync
        self.__verbose = verbose
        self.__epoch_range = epoch_range
        self.__batch_size_range = batch_size_range
        self.__lr_range = lr_range
        self.__momentum_range = momentum_range
        self.__train_ds = train_ds
        self.__val_ds = val_ds
        self.__test_ds = test_ds

    def training(self) -> None:
        top_1_metrics = TOP_1 if self.__dataset == "imagenet" else TOP_1_SPARSE
        top_5_metrics = TOP_5 if self.__dataset == "imagenet" else TOP_5_SPARSE
        loss_function = LOSS if self.__dataset == "imagenet" else LOSS_SPARSE

        for test_epoch in self.__epoch_range:
            for test_batch_size in self.__batch_size_range:
                for test_lr in self.__lr_range:
                    for test_momentum in self.__momentum_range:
                        backend.clear_session()
                        test_batch_size *= self.__num_device
                        print(test_epoch, test_batch_size, test_lr, test_momentum)
                        train_ds = self.__train_ds.batch(test_batch_size)
                        val_ds = self.__val_ds.batch(test_batch_size)
                        test_ds = self.__test_ds.batch(test_batch_size)

                        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
                        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
                        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

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
                            "./grid_search/"
                            + self.__dataset
                            + "/epoch"
                            + str(test_epoch)
                            + "batch_size"
                            + str(test_batch_size / self.__num_device)
                            + "lr"
                            + str(test_lr)
                            + "momentum"
                            + str(test_momentum)
                            + ".csv",
                        )

                        with self.__distributed_strategy.scope():
                            cnn = VGG16(
                                input_shape=self.__input_shape,
                                num_class=self.__num_classes,
                            )
                            cnn.summary()

                            cnn.compile(
                                optimizer=optimizers.SGD(
                                    learning_rate=test_lr, momentum=test_momentum
                                ),
                                loss=loss_function,
                                metrics=["accuracy", top_1_metrics, top_5_metrics],
                            )

                        start = time.time()
                        cnn.fit(
                            train_ds,
                            batch_size=test_batch_size,
                            epochs=test_epoch,
                            verbose=self.__verbose,
                            validation_data=(val_ds),
                        )
                        end = time.time()
                        time_taken = end - start

                        metrics = cnn.evaluate(
                            test_ds,
                            batch_size=test_batch_size,
                            return_dict=True,
                        )

                        accuracy = metrics["accuracy"]
                        top_1 = metrics["Top_1"]
                        top_5 = metrics["Top_5"]

                        print(f"Time taken: {time_taken}, Accuracy: {accuracy}")
                        write_csv(
                            [
                                {
                                    "test_epoch": test_epoch,
                                    "test_batch_size": test_batch_size
                                    / self.__num_device,
                                    "test_lr": test_lr,
                                    "test_momentum": test_momentum,
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
                            "./grid_search/"
                            + self.__dataset
                            + "/epoch"
                            + str(test_epoch)
                            + "batch_size"
                            + str(test_batch_size / self.__num_device)
                            + "lr"
                            + str(test_lr)
                            + "momentum"
                            + str(test_momentum)
                            + ".csv",
                        )
