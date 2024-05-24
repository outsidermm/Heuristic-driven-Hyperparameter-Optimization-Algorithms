import tensorflow as tf
from keras import datasets, optimizers,losses, callbacks
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from utility.AlexNet import AlexNet
import time

(X_train, y_train), (X_test, y_test) = datasets.cifar100.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, shuffle=True)

IMG_HEIGHT = IMG_WIDTH = 32
NUM_CLASSES :int = 100
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Make column vector into row vector
y_train = y_train.reshape(-1,)


def plot_sample(X, y, index):
    plt.imshow(X[index])
    plt.xlabel(y[index])
    plt.show()


# Normalize RGB to 0-1
X_train /= 255.0
X_test /= 255.0
X_val /= 255.0
cnn = AlexNet(INPUT_SHAPE, NUM_CLASSES)

start = time.time()
cnn.compile(
    optimizer=optimizers.legacy.SGD(), loss=losses.SparseCategoricalCrossentropy(), metrics=["accuracy"]
)

early_stopping = callbacks.EarlyStopping(monitor='loss', patience=5)
cnn.fit(X_train, y_train, batch_size=64, epochs=1, verbose=1, validation_data=(X_val,y_val),callbacks=[early_stopping])
end = time.time()
time_taken = end - start

metrics = cnn.evaluate(X_test, y_test)
accuracy = metrics[1]

print (f"Time taken: {time_taken}, Accuracy: {accuracy}")
