import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

(X_train, y_train), (X_test, y_test) = datasets.cifar100.load_data()

# Make column vector into row vector
y_train = y_train.reshape(
    -1,
)


def plot_sample(X, y, index):
    plt.imshow(X[index])
    plt.xlabel(y[index])
    plt.show()


# Normalize RGB to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

cnn = models.Sequential(
    [
        # CNN Layers
        layers.Conv2D(
            filters=32, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3)
        ),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(
            filters=32, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3)
        ),
        layers.MaxPooling2D((2,2)),
        # Dense Network
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(100, activation="softmax"),
    ]
)

cnn.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

cnn.fit(X_train, y_train, epochs=5)

cnn.evaluate(X_test, y_test)

y_pred = cnn.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print(classification_report(y_test, y_pred_classes))
