from keras import layers, models


def preprocessing() -> models.Sequential:
    preprocessing = models.Sequential()

    # Preprocessing - Data Augmentation
    preprocessing.add(layers.RandomBrightness(factor=0.2, value_range=(0, 1), seed=42))
    preprocessing.add(layers.RandomContrast(factor=0.2, seed=42))
    preprocessing.add(layers.RandomRotation(factor=0.2, seed=42))
    preprocessing.add(
        layers.RandomZoom(height_factor=(0.2, 0.2), width_factor=(0.2, 0.2), seed=42)
    )
    preprocessing.add(layers.RandomFlip(seed=42))

    return preprocessing
