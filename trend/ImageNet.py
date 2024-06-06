import sys

sys.path.append(".")

from keras import optimizers, losses, callbacks, layers, models, applications
from sklearn.model_selection import train_test_split
from utility.Model import preprocessing
from utility.DataAnalysis import write_csv, write_header
import time
import numpy as np

X_train: np.ndarray = np.load("./dataset/ImageNet/X_train.npy")
y_train: np.ndarray = np.load("./dataset/ImageNet/y_train.npy")
X_test: np.ndarray = np.load("./dataset/ImageNet/X_test.npy")
y_test: np.ndarray = np.load("./dataset/ImageNet/y_test.npy")


X_train = applications.resnet_v2.preprocess_input(X_train)
X_test = applications.resnet_v2.preprocess_input(X_test)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.20, shuffle=True, random_state=42
)


IMG_HEIGHT = IMG_WIDTH = 64
NUM_CLASSES: int = 200
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

early_stopping = callbacks.EarlyStopping(monitor="loss", patience=5)

### EPOCHS ###

write_header(["Epochs", "Time", "Accuracy"], "./trend_graph/ImageNet/epochs.csv")
for epochs in range(10, 210, 10):
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
    print(cnn.summary())
    start = time.time()
    cnn.compile(
        optimizer=optimizers.SGD(),
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    cnn.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=epochs,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )
    end = time.time()
    time_taken = end - start

    metrics = cnn.evaluate(X_test, y_test)
    accuracy = metrics[1]

    print(f"Time taken: {time_taken}, Accuracy: {accuracy}")
    write_csv(
        [{"Epochs": epochs, "Time": time_taken, "Accuracy": accuracy}],
        ["Epochs", "Time", "Accuracy"],
        "./trend_graph/ImageNet/epochs.csv",
    )

### BATCH SIZE ###

write_header(
    ["Batch_size", "Time", "Accuracy"], "./trend_graph/ImageNet/batch_size.csv"
)
for batch_size_power in range(4, 10):
    batch_size = 2**batch_size_power
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

    start = time.time()
    cnn.compile(
        optimizer=optimizers.SGD(),
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    cnn.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=75,
        verbose=2,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )
    end = time.time()
    time_taken = end - start

    metrics = cnn.evaluate(X_test, y_test)
    accuracy = metrics[1]

    print(f"Time taken: {time_taken}, Accuracy: {accuracy}")
    write_csv(
        [{"Batch_size": batch_size, "Time": time_taken, "Accuracy": accuracy}],
        ["Epochs", "Time", "Accuracy"],
        "./trend_graph/ImageNet/batch_size.csv",
    )


### LEARNING RATE ###
write_header(
    ["learning_rate", "Time", "Accuracy"], "./trend_graph/ImageNet/learning_rate.csv"
)
for learning_rate_power in np.linspace(1,6.5,11):
    learning_rate = 10**-learning_rate_power
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

    start = time.time()
    cnn.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate),
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    cnn.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=75,
        verbose=2,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )
    end = time.time()
    time_taken = end - start

    metrics = cnn.evaluate(X_test, y_test)
    accuracy = metrics[1]

    print(f"Time taken: {time_taken}, Accuracy: {accuracy}")
    write_csv(
        [{"learning_rate": learning_rate, "Time": time_taken, "Accuracy": accuracy}],
        ["learning_rate", "Time", "Accuracy"],
        "./trend_graph/ImageNet/learning_rate.csv",
    )


### MOMENTUM ###
write_header(["momentum", "Time", "Accuracy"], "./trend_graph/ImageNet/momentum.csv")
for momentum in np.linspace(0,0.95,0.05):
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

    start = time.time()
    cnn.compile(
        optimizer=optimizers.SGD(momentum=momentum),
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    cnn.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=75,
        verbose=2,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )
    end = time.time()
    time_taken = end - start

    metrics = cnn.evaluate(X_test, y_test)
    accuracy = metrics[1]

    print(f"Time taken: {time_taken}, Accuracy: {accuracy}")
    write_csv(
        [{"momentum": momentum, "Time": time_taken, "Accuracy": accuracy}],
        ["momentum", "Time", "Accuracy"],
        "./trend_graph/ImageNet/momentum.csv",
    )
