from keras import layers, models


def data_augmentation() -> models.Sequential:
    preprocessing = models.Sequential()
    # Preprocessing - Data Augmentation
    preprocessing.add(layers.RandomContrast(factor=0.5, seed=42))
    preprocessing.add(layers.RandomFlip(seed=42))

    return preprocessing


def normalisation() -> models.Sequential:
    normalisation = models.Sequential()
    normalisation.add(layers.Rescaling(scale=1.0 / 127.5, offset=-1))
    return normalisation


def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = layers.Conv2D(filter, (3, 3), padding="same")(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation("relu")(x)
    # Layer 2
    x = layers.Conv2D(filter, (3, 3), padding="same")(x)
    x = layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = layers.Add()([x, x_skip])
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    return x


def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = layers.Conv2D(filter, (3, 3), padding="same", strides=(2, 2))(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation("relu")(x)
    # Layer 2
    x = layers.Conv2D(filter, (3, 3), padding="same")(x)
    x = layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = layers.Conv2D(filter, (1, 1), strides=(2, 2))(x_skip)
    # Add Residue
    x = layers.Add()([x, x_skip])
    x = layers.Activation("relu")(x)
    return x


def ResNet18(shape=(32, 32, 3), classes=10):
    # Step 1 (Setup Input Layer)
    x_input = layers.Input(shape)
    x = layers.ZeroPadding2D((3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [2, 2, 2, 2]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size * 2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = layers.AveragePooling2D((2, 2), padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(classes, activation="softmax")(x)
    model = models.Model(inputs=x_input, outputs=x, name="ResNet34")
    return model
