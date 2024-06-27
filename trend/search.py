from hyperparameter_search import HyperParameterSearch
import utility.const as const

if __name__ == "__main__":
    cifar_100_epoch = HyperParameterSearch(
        "CIFAR-100",
        hyperparameter="epoch",
        img_height=const.CIFAR_IMG_HEIGHT,
        img_width=const.CIFAR_IMG_WIDTH,
        num_classes=const.CIFAR_NUM_CLASSES,
        epoch=const.STD_EPOCH,
        batch_size=const.STD_BATCH_SIZE,
        lr=const.STD_LR,
        momentum=const.STD_MOMENTUM,
        verbose=1,
    )
    cifar_100_epoch.load_dataset()
    cifar_100_epoch.training()

    image_net_epoch = HyperParameterSearch(
        "imagenet",
        hyperparameter="epoch",
        img_height=const.IMAGENET_IMG_HEIGHT,
        img_width=const.IMAGENET_IMG_WIDTH,
        num_classes=const.IMAGENET_NUM_CLASSES,
        epoch=const.STD_EPOCH,
        batch_size=const.STD_BATCH_SIZE,
        lr=const.STD_LR,
        momentum=const.STD_MOMENTUM,
        verbose=1,
    )
    image_net_epoch.load_dataset()
    image_net_epoch.training()
