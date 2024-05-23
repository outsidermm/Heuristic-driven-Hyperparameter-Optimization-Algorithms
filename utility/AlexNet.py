from keras import layers, models

def AlexNet (input_shape, num_classes: int) -> models.Sequential:
    AlexNet = models.Sequential()

    AlexNet.add(layers.Resizing(224, 224, input_shape=input_shape))

    #Preprocessing - Data Augmentation
    AlexNet.add(layers.RandomBrightness(factor=0.2,value_range=(0,1)))
    AlexNet.add(layers.RandomContrast(factor=0.2))
    AlexNet.add(layers.RandomRotation(factor=0.2))
    AlexNet.add(layers.RandomZoom(height_factor=(0.2,0.2), width_factor=(0.2,0.2)))

    #Convolutional Layers
    AlexNet.add(layers.Conv2D(filters=96,kernel_size=(3,3),strides=(4,4), padding='same'))
    AlexNet.add(layers.BatchNormalization())
    AlexNet.add(layers.Activation('relu'))
    AlexNet.add(layers.MaxPooling2D(pool_size=(3,3), strides=2))

    AlexNet.add(layers.Conv2D(filters=256, kernel_size=5, strides=(4,4), padding='same'))
    AlexNet.add(layers.BatchNormalization())
    AlexNet.add(layers.Activation('relu'))
    AlexNet.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    AlexNet.add(layers.Conv2D(filters=384, kernel_size=3, strides=(4,4), padding='same', activation='relu'))
    AlexNet.add(layers.Conv2D(filters=384, kernel_size=3, strides=(4,4), padding='same', activation='relu'))
    AlexNet.add(layers.Conv2D(filters=256, kernel_size=3, strides=(4,4), padding='same', activation='relu'))
    AlexNet.add(layers.Flatten())

    #Dense Layers
    AlexNet.add(layers.Dense(4096, activation='relu'))
    AlexNet.add(layers.Dropout(0.5))
    AlexNet.add(layers.Dense(4096, activation='relu'))
    AlexNet.add(layers.Dropout(0.5))
    
    #Resultant
    AlexNet.add(layers.Dense(num_classes, activation='softmax'))

    print(AlexNet.summary())
    return AlexNet