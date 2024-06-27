from hyperparameter_search import HyperParameterSearch

STD_EPOCH = 350
STD_BATCH_SIZE = 32
STD_LR = 0.01
STD_MOMENTUM = 0.9

IMAGENET_IMG_HEIGHT = 64
IMAGENET_IMG_WIDTH = 64
IMAGENET_NUM_CLASSES = 200

CIFAR_IMG_HEIGHT = 32
CIFAR_IMG_WIDTH = 32
CIFAR_NUM_CLASSES = 100

if __name__ == "__main__":
    cifar_100_epoch = HyperParameterSearch(
        "CIFAR-100",
        hyperparameter="epoch",
        img_height=CIFAR_IMG_HEIGHT,
        img_width=CIFAR_IMG_WIDTH,
        num_classes=CIFAR_NUM_CLASSES,
        epoch=STD_EPOCH,
        batch_size=STD_BATCH_SIZE,
        lr=STD_LR,
        momentum=STD_MOMENTUM,
        verbose=1,
    )
    cifar_100_epoch.load_dataset()
    cifar_100_epoch.training()

    image_net_epoch = HyperParameterSearch(
        "imagenet",
        hyperparameter="epoch",
        img_height=IMAGENET_IMG_HEIGHT,
        img_width=IMAGENET_IMG_WIDTH,
        num_classes=IMAGENET_NUM_CLASSES,
        epoch=STD_EPOCH,
        batch_size=STD_BATCH_SIZE,
        lr=STD_LR,
        momentum=STD_MOMENTUM,
        verbose=1,
    )
    image_net_epoch.load_dataset()
    image_net_epoch.training()