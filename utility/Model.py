from keras import layers, models, regularizers


def data_augmentation() -> models.Sequential:
    preprocessing = models.Sequential()
    # Preprocessing - Data Augmentation
    preprocessing.add(layers.RandomContrast(factor=0.2, seed=42))
    preprocessing.add(layers.RandomFlip(mode="HORIZONTAL", seed=42))
    preprocessing.add(layers.RandomRotation(factor=0.1, seed=42))
    preprocessing.add(
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=42)
    )

    return preprocessing


def normalisation() -> models.Sequential:
    normalisation = models.Sequential()
    normalisation.add(layers.Rescaling(scale=1.0 / 255))
    return normalisation


def add_layer(model, num, dropout=True, weight_decay=0.0005):
    model.add(
        layers.Conv2D(
            num,
            (3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
    )
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())
    if dropout:
        model.add(layers.Dropout(0.4))
    return model


def VGG16(input_shape, num_class, weight_decay=0.0005):

    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(
        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
    )
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model = add_layer(model, 64, dropout=False)

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model = add_layer(model, 128, dropout=True)

    model = add_layer(model, 128, dropout=False)

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model = add_layer(model, 256, dropout=True)

    model = add_layer(model, 256, dropout=True)

    model = add_layer(model, 256, dropout=False)

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model = add_layer(model, 512, dropout=True)

    model = add_layer(model, 512, dropout=True)

    model = add_layer(model, 512, dropout=False)

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model = add_layer(model, 512, dropout=True)

    model = add_layer(model, 512, dropout=True)

    model = add_layer(model, 512, dropout=False)

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_class))
    model.add(layers.Activation("softmax"))

    return model
