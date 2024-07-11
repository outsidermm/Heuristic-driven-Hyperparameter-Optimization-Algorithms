from hyperparameter_search import HyperParameterSearch
from utility.dataloader import DataLoader

if __name__ == "__main__":
    cifar_100_data = DataLoader("CIFAR-100")
    cifar_100_train_ds, cifar_100_val_ds, cifar_100_test_ds = cifar_100_data.load_dataset()

    cifar_100_lr = HyperParameterSearch(
        "CIFAR-100",
        "lr",
        train_ds=cifar_100_train_ds,
        val_ds=cifar_100_val_ds,
        test_ds=cifar_100_test_ds,
        verbose=1,
    )

    cifar_100_lr.training()

    cifar_100_batch_size = HyperParameterSearch(
        "CIFAR-100",
        "batch_size",
        train_ds=cifar_100_train_ds,
        val_ds=cifar_100_val_ds,
        test_ds=cifar_100_test_ds,
        verbose=1,
    )
    cifar_100_batch_size.training()



