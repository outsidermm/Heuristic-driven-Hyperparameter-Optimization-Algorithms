from hyperparameter_search import HyperParameterSearch
import utility.const as const

if __name__ == "__main__":
    image_net_batch = HyperParameterSearch(
        "imagenet",
        hyperparameter="batch_size",
        img_height=const.IMAGENET_IMG_HEIGHT,
        img_width=const.IMAGENET_IMG_WIDTH,
        num_classes=const.IMAGENET_NUM_CLASSES,
        epoch=const.STD_EPOCH,
        batch_size=const.STD_BATCH_SIZE,
        lr=const.STD_LR,
        momentum=const.STD_MOMENTUM,
        verbose=1,
    )
    image_net_batch.load_dataset()
    image_net_batch.training()

    image_net_lr = HyperParameterSearch(
        "imagenet",
        hyperparameter="lr",
        img_height=const.IMAGENET_IMG_HEIGHT,
        img_width=const.IMAGENET_IMG_WIDTH,
        num_classes=const.IMAGENET_NUM_CLASSES,
        epoch=const.STD_EPOCH,
        batch_size=const.STD_BATCH_SIZE,
        lr=const.STD_LR,
        momentum=const.STD_MOMENTUM,
        verbose=1,
    )
    image_net_lr.load_dataset()
    image_net_lr.training()

    image_net_momentum = HyperParameterSearch(
        "imagenet",
        hyperparameter="momentum",
        img_height=const.IMAGENET_IMG_HEIGHT,
        img_width=const.IMAGENET_IMG_WIDTH,
        num_classes=const.IMAGENET_NUM_CLASSES,
        epoch=const.STD_EPOCH,
        batch_size=const.STD_BATCH_SIZE,
        lr=const.STD_LR,
        momentum=const.STD_MOMENTUM,
        verbose=1,
    )
    image_net_momentum.load_dataset()
    image_net_momentum.training()
