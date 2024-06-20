from hyperparameter_search import HyperParameterSearch

STD_EPOCH = 50
STD_BATCH_SIZE = 32
STD_LR = 0.01
STD_MOMENTUM = 0.0

IMAGENET_IMG_HEIGHT = 64
IMAGENET_IMG_WIDTH = 64
IMAGENET_NUM_CLASSES = 200

CIFAR_IMG_HEIGHT = 32
CIFAR_IMG_WIDTH = 32
CIFAR_NUM_CLASSES = 100

if __name__ == "__main__":
    imagenet_epoch = HyperParameterSearch(
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
    imagenet_epoch.load_dataset()
    imagenet_epoch.training()
