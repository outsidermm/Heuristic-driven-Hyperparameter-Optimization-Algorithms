from grid_search import GridSearch
from utility.dataloader import DataLoader

if __name__ == "__main__":
    cifar_100_data = DataLoader("CIFAR-100")
    cifar_100_train_ds, cifar_100_val_ds, cifar_100_test_ds = cifar_100_data.load_dataset()

    cifar_100 = GridSearch(
        "CIFAR-100",
        train_ds=cifar_100_train_ds,
        val_ds=cifar_100_val_ds,
        test_ds=cifar_100_test_ds,
        verbose=1,
    )
    cifar_100.training()
