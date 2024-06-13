import sys

sys.path.append(".")

from keras import optimizers, losses, callbacks, models, layers, applications
from sklearn.model_selection import train_test_split
from utility.Model import preprocessing
from utility.DataAnalysis import write_csv, write_header
import time
import numpy as np
import tensorflow as tf

X_train: np.ndarray = np.load("./dataset/CIFAR-100/X_train.npy")
y_train: np.ndarray = np.load("./dataset/CIFAR-100/y_train.npy")
X_test: np.ndarray = np.load("./dataset/CIFAR-100/X_test.npy")
y_test: np.ndarray = np.load("./dataset/CIFAR-100/y_test.npy")

X_train = applications.resnet_v2.preprocess_input(X_train)
X_test = applications.resnet_v2.preprocess_input(X_test)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.20, shuffle=True, random_state=42
)

IMG_HEIGHT = IMG_WIDTH = 32
NUM_CLASSES: int = 100
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

distributed_strategy = tf.distribute.MirroredStrategy()
NUM_DEVICE = distributed_strategy.num_replicas_in_sync

STD_EPOCH = 500
STD_BATCH_SIZE = 32 * NUM_DEVICE

### EPOCHS ###
write_header(["Epochs", "Time", "Accuracy"], "./trend_graph/CIFAR-100/epochs.csv")
for epochs in range(100, 1601, 300):
    with distributed_strategy.scope():
        cnn = models.Sequential(
            [
                layers.InputLayer(INPUT_SHAPE),
                layers.Resizing(224, 224),
                preprocessing(),
                applications.ResNet50V2(include_top=False),
                layers.GlobalMaxPooling2D(),
                layers.Dense(NUM_CLASSES, activation="softmax"),
            ]
        )

        cnn.compile(
            optimizer=optimizers.SGD(),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    start = time.time()
    cnn.fit(
        X_train,
        y_train,
        batch_size=STD_BATCH_SIZE,
        epochs=epochs,
        verbose=2,
        validation_data=(X_val, y_val),
    )
    end = time.time()
    time_taken = end - start

    metrics = cnn.evaluate(X_test, y_test)
    accuracy = metrics[1]

    print(f"Time taken: {time_taken}, Accuracy: {accuracy}")
    write_csv(
        [{"Epochs": epochs, "Time": time_taken, "Accuracy": accuracy}],
        ["Epochs", "Time", "Accuracy"],
        "./trend_graph/CIFAR-100/epochs.csv",
    )

### BATCH SIZE ###

write_header(
    ["Batch_size", "Time", "Accuracy"], "./trend_graph/CIFAR-100/batch_size.csv"
)
for batch_size_power in range(3, 7):
    batch_size = 2 ** (batch_size_power)
    batch_size = batch_size * NUM_DEVICE

    with distributed_strategy.scope():
        cnn = models.Sequential(
            [
                layers.InputLayer(INPUT_SHAPE),
                layers.Resizing(224, 224),
                preprocessing(),
                applications.ResNet50V2(include_top=False),
                layers.GlobalMaxPooling2D(),
                layers.Dense(NUM_CLASSES, activation="softmax"),
            ]
        )

        cnn.compile(
            optimizer=optimizers.SGD(),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    start = time.time()

    cnn.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=STD_EPOCH,
        verbose=2,
        validation_data=(X_val, y_val),
    )
    end = time.time()
    time_taken = end - start

    metrics = cnn.evaluate(X_test, y_test)
    accuracy = metrics[1]

    print(f"Time taken: {time_taken}, Accuracy: {accuracy}")
    write_csv(
        [{"Batch_size": batch_size, "Time": time_taken, "Accuracy": accuracy}],
        ["Batch_size", "Time", "Accuracy"],
        "./trend_graph/CIFAR-100/batch_size.csv",
    )


### LEARNING RATE ###
write_header(
    ["learning_rate", "Time", "Accuracy"], "./trend_graph/CIFAR-100/learning_rate.csv"
)
for learning_rate_power in range(1, 7):
    learning_rate = 10**-learning_rate_power

    with distributed_strategy.scope():
        cnn = models.Sequential(
            [
                layers.InputLayer(INPUT_SHAPE),
                layers.Resizing(224, 224),
                preprocessing(),
                applications.ResNet50V2(include_top=False),
                layers.GlobalMaxPooling2D(),
                layers.Dense(NUM_CLASSES, activation="softmax"),
            ]
        )

        cnn.compile(
            optimizer=optimizers.SGD(),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    start = time.time()

    cnn.fit(
        X_train,
        y_train,
        batch_size=STD_BATCH_SIZE,
        epochs=STD_EPOCH,
        verbose=2,
        validation_data=(X_val, y_val),
    )
    end = time.time()
    time_taken = end - start

    metrics = cnn.evaluate(X_test, y_test)
    accuracy = metrics[1]

    print(f"Time taken: {time_taken}, Accuracy: {accuracy}")
    write_csv(
        [{"learning_rate": learning_rate, "Time": time_taken, "Accuracy": accuracy}],
        ["learning_rate", "Time", "Accuracy"],
        "./trend_graph/CIFAR-100/learning_rate.csv",
    )


### MOMENTUM ###
write_header(["momentum", "Time", "Accuracy"], "./trend_graph/CIFAR-100/momentum.csv")
for momentum in np.linspace(0, 0.9, 7):
    with distributed_strategy.scope():
        cnn = models.Sequential(
            [
                layers.InputLayer(INPUT_SHAPE),
                layers.Resizing(224, 224),
                preprocessing(),
                applications.ResNet50V2(include_top=False),
                layers.GlobalMaxPooling2D(),
                layers.Dense(NUM_CLASSES, activation="softmax"),
            ]
        )

        cnn.compile(
            optimizer=optimizers.SGD(),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    start = time.time()

    cnn.fit(
        X_train,
        y_train,
        batch_size=STD_BATCH_SIZE,
        epochs=STD_EPOCH,
        verbose=2,
        validation_data=(X_val, y_val),
    )
    end = time.time()
    time_taken = end - start

    metrics = cnn.evaluate(X_test, y_test)
    accuracy = metrics[1]

    print(f"Time taken: {time_taken}, Accuracy: {accuracy}")
    write_csv(
        [{"momentum": momentum, "Time": time_taken, "Accuracy": accuracy}],
        ["momentum", "Time", "Accuracy"],
        "./trend_graph/CIFAR-100/momentum.csv",
    )
