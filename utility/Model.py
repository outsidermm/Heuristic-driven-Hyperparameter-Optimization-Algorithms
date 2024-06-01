from keras import layers, models


def preprocessing() -> models.Sequential:
    preprocessing = models.Sequential()

    # Preprocessing - Data Augmentation
    preprocessing.add(layers.RandomBrightness(factor=0.2, value_range=(0, 1)))
    preprocessing.add(layers.RandomContrast(factor=0.2))
    preprocessing.add(layers.RandomRotation(factor=0.2))
    preprocessing.add(
        layers.RandomZoom(height_factor=(0.2, 0.2), width_factor=(0.2, 0.2))
    )
    preprocessing.add(layers.RandomFlip())

    # Preprocessing - Batch Normalisation
    preprocessing.add(layers.BatchNormalization())

    return preprocessing


def alex_net() -> models.Sequential:
    alex_net = models.Sequential()

    # Convolutional Layers
    alex_net.add(
        layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(4, 4), padding="same")
    )
    alex_net.add(layers.BatchNormalization())
    alex_net.add(layers.Activation("relu"))
    alex_net.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    alex_net.add(
        layers.Conv2D(filters=256, kernel_size=5, strides=(4, 4), padding="same")
    )
    alex_net.add(layers.BatchNormalization())
    alex_net.add(layers.Activation("relu"))
    alex_net.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    alex_net.add(
        layers.Conv2D(
            filters=384,
            kernel_size=3,
            strides=(4, 4),
            padding="same",
            activation="relu",
        )
    )
    alex_net.add(
        layers.Conv2D(
            filters=384,
            kernel_size=3,
            strides=(4, 4),
            padding="same",
            activation="relu",
        )
    )
    alex_net.add(
        layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=(4, 4),
            padding="same",
            activation="relu",
        )
    )
    alex_net.add(layers.Flatten())

    # Dense Layers
    alex_net.add(layers.Dense(4096, activation="relu"))
    alex_net.add(layers.Dropout(0.5))
    alex_net.add(layers.Dense(4096, activation="relu"))
    alex_net.add(layers.Dropout(0.5))

    return alex_net
