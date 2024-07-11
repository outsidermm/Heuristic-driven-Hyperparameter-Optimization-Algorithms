from typing import Tuple
from keras import layers, models, regularizers


def normalisation() -> models.Sequential:
    normalisation = models.Sequential()
    normalisation.add(layers.Rescaling(scale=1.0 / 255))
    return normalisation


def add_layer(
    model: models.Sequential,
    filter_num: int,
    dropout: bool = True,
    weight_decay: float = 0.0005,
) -> models.Sequential:
    model.add(
        layers.Conv2D(
            filter_num,
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


def VGG16(
    input_shape: Tuple[int, int, int], num_class: int, weight_decay: float = 0.0005
):

    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # Preprocessing - Data Augmentation
    model.add(layers.RandomFlip(mode="HORIZONTAL", seed=42))
    model.add(layers.RandomRotation(factor=0.1, seed=42))
    model.add(layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=42))

    # Preprocessing - VGG-16
    model.add(
        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
    )
    model.add(layers.ReLU())
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
    model.add(layers.ReLU())
    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_class))
    model.add(layers.Softmax())

    return model
