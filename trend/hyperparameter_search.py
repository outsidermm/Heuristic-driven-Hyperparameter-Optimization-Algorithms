import sys

sys.path.append(".")

from keras import optimizers, losses, metrics, callbacks, backend
from sklearn.model_selection import train_test_split
from utility.Model import data_augmentation, VGG16, normalisation
from utility.DataAnalysis import write_csv, write_header
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
        img_height: int,
        img_width: int,
        num_classes: int,
        epoch: int = 500,
        batch_size: int = 32,
        lr: float = 0.01,
        momentum: float = 0.0,
        verbose: int = 2,
    ) -> None:
        self.__dataset = dataset
        self.__hyperparameter = hyperparameter
        self.__img_height = img_height
        self.__img_width = img_width
        self.__num_classes = num_classes
        self.__input_shape = (img_height, img_width, 3)
        self.__epoch = epoch
        self.__batch_size = batch_size
        self.__momentum = momentum
        self.__lr = lr
        self.__distributed_strategy = tf.distribute.MirroredStrategy()
        self.__num_device = self.__distributed_strategy.num_replicas_in_sync
        self.__batch_size *= self.__num_device
        self.__verbose = verbose

    def load_dataset(self) -> None:
        self.__X_train: np.ndarray = np.load(
            "./dataset/" + self.__dataset + "/X_train.npy"
        )
        self.__y_train: np.ndarray = np.load(
            "./dataset/" + self.__dataset + "/y_train.npy"
        )
        self.__X_test: np.ndarray = np.load(
            "./dataset/" + self.__dataset + "/X_test.npy"
        )
        self.__y_test: np.ndarray = np.load(
            "./dataset/" + self.__dataset + "/y_test.npy"
        )

        self.__X_train, self.__X_val, self.__y_train, self.__y_val = train_test_split(
            self.__X_train,
            self.__y_train,
            test_size=0.40,
            random_state=42,
        )

        self.__X_train = normalisation()(self.__X_train)
        self.__X_val = normalisation()(self.__X_val)
        self.__X_test = normalisation()(self.__X_test)

        data_augmentation_layer = data_augmentation()
        augmented_images = data_augmentation_layer(self.__X_train)

        self.__X_train = np.concatenate([self.__X_train, augmented_images], axis=0)
        self.__y_train = np.concatenate([self.__y_train, self.__y_train], axis=0)

        self.__train_ds = tf.data.Dataset.from_tensor_slices(
            (self.__X_train, self.__y_train)
        )
        self.__val_ds = tf.data.Dataset.from_tensor_slices((self.__X_val, self.__y_val))
        self.__test_ds = tf.data.Dataset.from_tensor_slices(
            (self.__X_test, self.__y_test)
        )

    def training(self) -> None:
        top_1_metrics = TOP_1
        top_5_metrics = TOP_5
        loss_function = LOSS
        if self.__dataset == "CIFAR-100":
            top_1_metrics = TOP_1_SPARSE
            top_5_metrics = TOP_5_SPARSE
            loss_function = LOSS_SPARSE

        search_space = None
        if self.__hyperparameter == "epoch":
            search_space = np.arange(20, 381, step=90)
        elif self.__hyperparameter == "batch_size":
            linear_search_space = np.arange(3, 8)  # 3-7
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

        self.__train_ds = self.__train_ds.batch(test_batch_size)
        self.__val_ds = self.__val_ds.batch(test_batch_size)
        self.__test_ds = self.__test_ds.batch(test_batch_size)

        self.__train_ds = self.__train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.__val_ds = self.__val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.__test_ds = self.__test_ds.cache().prefetch(buffer_size=AUTOTUNE)

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
                self.__train_ds,
                batch_size=test_batch_size,
                epochs=test_epoch,
                verbose=self.__verbose,
                validation_data=(self.__val_ds),
                callbacks=[logger],
            )
            end = time.time()
            time_taken = end - start

            metrics = cnn.evaluate(
                self.__test_ds,
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
                "./trend/"
                + self.__dataset
                + "/"
                + self.__hyperparameter
                + ".csv",
            )
