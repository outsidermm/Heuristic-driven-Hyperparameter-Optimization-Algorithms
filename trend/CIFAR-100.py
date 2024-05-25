import sys

sys.path.append(".")

from keras import datasets, optimizers,losses, callbacks
from sklearn.model_selection import train_test_split
from utility.AlexNet import AlexNet
from utility.DataAnalysis import write_csv, read_csv, plot
import time

(X_train, y_train), (X_test, y_test) = datasets.cifar100.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, shuffle=True)

IMG_HEIGHT = IMG_WIDTH = 32
NUM_CLASSES :int = 100
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Make column vector into row vector
y_train = y_train.reshape(-1,)

# Normalize RGB to 0-1
X_train = X_train / 255
X_test = X_test / 255
X_val = X_val / 255

### EPOCHS ###

Hyperparameters:list[dict] = []

for epochs in range(10,200,5):
    cnn = AlexNet(INPUT_SHAPE, NUM_CLASSES)

    start = time.time()
    cnn.compile(
        optimizer=optimizers.SGD(), loss=losses.SparseCategoricalCrossentropy(), metrics=["accuracy"]
    )

    early_stopping = callbacks.EarlyStopping(monitor='loss', patience=5)
    cnn.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0, validation_data=(X_val,y_val),callbacks=[early_stopping])
    end = time.time()
    time_taken = end - start

    metrics = cnn.evaluate(X_test, y_test)
    accuracy = metrics[1]

    print (f"Time taken: {time_taken}, Accuracy: {accuracy}")
    Hyperparameters.append({"Epochs":epochs, "Time":time_taken, "Accuracy":accuracy})

write_csv(Hyperparameters,["Epochs","Time","Accuracy"],"./trend_graph/CIFAR-100/epochs.csv")

### BATCH SIZE ###

Hyperparameters:list[dict] = []

for batch_size_power in range(4,9):
    batch_size = 2**batch_size_power
    cnn = AlexNet(INPUT_SHAPE, NUM_CLASSES)

    start = time.time()
    cnn.compile(
        optimizer=optimizers.SGD(), loss=losses.SparseCategoricalCrossentropy(), metrics=["accuracy"]
    )

    early_stopping = callbacks.EarlyStopping(monitor='loss', patience=5)
    cnn.fit(X_train, y_train, batch_size=batch_size, epochs=75, verbose=0, validation_data=(X_val,y_val),callbacks=[early_stopping])
    end = time.time()
    time_taken = end - start

    metrics = cnn.evaluate(X_test, y_test)
    accuracy = metrics[1]

    print (f"Time taken: {time_taken}, Accuracy: {accuracy}")
    Hyperparameters.append({"Batch_size":batch_size, "Time":time_taken, "Accuracy":accuracy})

write_csv(Hyperparameters,["Batch_size","Time","Accuracy"],"./trend_graph/CIFAR-100/batch_size.csv")


### LEARNING RATE ###
Hyperparameters:list[dict] = []

for learning_rate_power in range(1,6,0.5):
    learning_rate = 10**-learning_rate_power
    cnn = AlexNet(INPUT_SHAPE, NUM_CLASSES)

    start = time.time()
    cnn.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate), loss=losses.SparseCategoricalCrossentropy(), metrics=["accuracy"]
    )

    early_stopping = callbacks.EarlyStopping(monitor='loss', patience=5)
    cnn.fit(X_train, y_train, batch_size=32, epochs=75, verbose=0, validation_data=(X_val,y_val),callbacks=[early_stopping])
    end = time.time()
    time_taken = end - start

    metrics = cnn.evaluate(X_test, y_test)
    accuracy = metrics[1]

    print (f"Time taken: {time_taken}, Accuracy: {accuracy}")
    Hyperparameters.append({"learning_rate":learning_rate, "Time":time_taken, "Accuracy":accuracy})

write_csv(Hyperparameters,["learning_rate","Time","Accuracy"],"./trend_graph/CIFAR-100/learning_rate.csv")


### MOMENTUM ###
Hyperparameters:list[dict] = []

for momentum in range(0,0.9,0.05):
    cnn = AlexNet(INPUT_SHAPE, NUM_CLASSES)

    start = time.time()
    cnn.compile(
        optimizer=optimizers.SGD(momentum=momentum), loss=losses.SparseCategoricalCrossentropy(), metrics=["accuracy"]
    )

    early_stopping = callbacks.EarlyStopping(monitor='loss', patience=5)
    cnn.fit(X_train, y_train, batch_size=32, epochs=75, verbose=0, validation_data=(X_val,y_val),callbacks=[early_stopping])
    end = time.time()
    time_taken = end - start

    metrics = cnn.evaluate(X_test, y_test)
    accuracy = metrics[1]

    print (f"Time taken: {time_taken}, Accuracy: {accuracy}")
    Hyperparameters.append({"momentum":momentum, "Time":time_taken, "Accuracy":accuracy})

write_csv(Hyperparameters,["momentum","Time","Accuracy"],"./trend_graph/CIFAR-100/momentum.csv")