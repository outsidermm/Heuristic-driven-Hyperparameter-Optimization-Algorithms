import sys

sys.path.append(".")

from keras import optimizers, losses, metrics, callbacks, backend
from utility.model import VGG16
from utility.helper_func import write_csv, write_header
import time
import numpy as np
import tensorflow as tf

TOP_1 = metrics.TopKCategoricalAccuracy(k=1, name="Top_1")
TOP_5 = metrics.TopKCategoricalAccuracy(k=5, name="Top_5")
TOP_1_SPARSE = metrics.SparseTopKCategoricalAccuracy(k=1, name="Top_1")
TOP_5_SPARSE = metrics.SparseTopKCategoricalAccuracy(k=5, name="Top_5")
LOSS = losses.CategoricalCrossentropy()
LOSS_SPARSE = losses.SparseCategoricalCrossentropy()
AUTOTUNE = tf.data.experimental.AUTOTUNE


class HyperParameterSearch:

    def __init__(
        self,
        dataset: str,
        hyperparameter: str,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        epoch: int = 250,
        batch_size: int = 128,
        lr: float = 0.01,
        momentum: float = 0.9,
        verbose: int = 2,
    ) -> None:
        self.__dataset = dataset
        self.__hyperparameter = hyperparameter
        self.__epoch = epoch
        self.__batch_size = batch_size
        self.__momentum = momentum
        self.__lr = lr
        self.__distributed_strategy = tf.distribute.MirroredStrategy()
        self.__num_device = self.__distributed_strategy.num_replicas_in_sync
        self.__batch_size *= self.__num_device
        self.__verbose = verbose
        self.__train_ds = train_ds
        self.__val_ds = val_ds
        self.__test_ds = test_ds
        self.__input_shape = (64, 64, 3) if dataset == "imagenet" else (32, 32, 3)
        self.__num_classes = 200 if self.__dataset == "imagenet" else 100
        backend.clear_session()

    def training(self) -> None:
        top_1_metrics = TOP_1 if self.__dataset == "imagenet" else TOP_1_SPARSE
        top_5_metrics = TOP_5 if self.__dataset == "imagenet" else TOP_5_SPARSE
        loss_function = LOSS if self.__dataset == "imagenet" else LOSS_SPARSE

        search_space = None
        if self.__hyperparameter == "epoch":
            search_space = np.arange(20, 381, step=90)
        elif self.__hyperparameter == "batch_size":
            linear_search_space = np.arange(4, 14, 2)  # 16 to 4096
            search_space = np.power(2, linear_search_space)
        elif self.__hyperparameter == "lr":
            linear_search_space = np.linspace(-7, -1, 7, endpoint=True)
            search_space = np.power(10, linear_search_space)
        elif self.__hyperparameter == "momentum":
            search_space = np.linspace(0, 0.9, 7)
        else:
            print("Wrong Hyperparameter Input!")

        test_epoch = self.__epoch
        test_batch_size = self.__batch_size
        test_lr = self.__lr
        test_momentum = self.__momentum

        write_header(
            [self.__hyperparameter, "Time", "Accuracy", "Top1", "Top5"],
            "./trend/" + self.__dataset + "/" + self.__hyperparameter + ".csv",
        )

        for changing_hp in search_space:
            backend.clear_session()
            if self.__hyperparameter == "epoch":
                test_epoch = changing_hp
            elif self.__hyperparameter == "batch_size":
                test_batch_size = changing_hp
            elif self.__hyperparameter == "lr":
                test_lr = changing_hp
            elif self.__hyperparameter == "momentum":
                test_momentum = changing_hp
            else:
                print("Wrong Hyperparameter Input!")

            train_ds = self.__train_ds.batch(test_batch_size)
            val_ds = self.__val_ds.batch(test_batch_size)
            test_ds = self.__test_ds.batch(test_batch_size)

            train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
            test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

            with self.__distributed_strategy.scope():
                cnn = VGG16(
                    input_shape=self.__input_shape, num_class=self.__num_classes
                )
                cnn.summary()

                cnn.compile(
                    optimizer=optimizers.SGD(
                        learning_rate=test_lr, momentum=test_momentum
                    ),
                    loss=loss_function,
                    metrics=["accuracy", top_1_metrics, top_5_metrics],
                )

            log = open(
                "./trend/"
                + self.__dataset
                + "/"
                + self.__hyperparameter
                + "_"
                + str(changing_hp)
                + "_log.csv",
                "a",
            )
            log.close()

            logger = callbacks.CSVLogger(
                "./trend/"
                + self.__dataset
                + "/"
                + self.__hyperparameter
                + "_"
                + str(changing_hp)
                + "_log.csv"
            )

            start = time.time()
            cnn.fit(
                train_ds,
                batch_size=test_batch_size,
                epochs=test_epoch,
                verbose=self.__verbose,
                validation_data=(val_ds),
                callbacks=[logger],
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
                        self.__hyperparameter: changing_hp,
                        "Time": time_taken,
                        "Accuracy": accuracy,
                        "Top1": top_1,
                        "Top5": top_5,
                    }
                ],
                [self.__hyperparameter, "Time", "Accuracy", "Top1", "Top5"],
                "./trend/" + self.__dataset + "/" + self.__hyperparameter + ".csv",
            )
