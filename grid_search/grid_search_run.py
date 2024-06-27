from grid_search import GridSearch
import const

if __name__ == "__main__":
    cifar_100 = GridSearch(
        "CIFAR-100",
        img_height=const.CIFAR_IMG_HEIGHT,
        img_width=const.CIFAR_IMG_WIDTH,
        num_classes=const.CIFAR_NUM_CLASSES,
        verbose=1,
    )
    cifar_100.load_dataset()
    cifar_100.training()

    image_net = GridSearch(
        "imagenet",
        img_height=const.IMAGENET_IMG_HEIGHT,
        img_width=const.IMAGENET_IMG_WIDTH,
        num_classes=const.IMAGENET_NUM_CLASSES,
        verbose=1,
    )
    image_net.load_dataset()
    image_net.training()
