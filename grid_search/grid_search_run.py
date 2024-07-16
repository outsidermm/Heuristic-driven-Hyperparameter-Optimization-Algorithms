from grid_search import GridSearch
from utility.dataloader import DataLoader

if __name__ == "__main__":
    imagenet_data = DataLoader("imagenet")
    imagenet_train_ds, imagenet_val_ds,imagenet_test_ds = imagenet_data.load_dataset()

    cifar_100 = GridSearch(
        "CIFAR-100",
        train_ds=imagenet_train_ds,
        val_ds=imagenet_val_ds,
        test_ds=imagenet_test_ds,
        verbose=1,
    )
    cifar_100.training()
