from hyperparameter_search import HyperParameterSearch
from utility.dataloader import DataLoader

if __name__ == "__main__":
    data = DataLoader("imagenet")
    train_ds, val_ds, test_ds = data.load_dataset()

    imagenet_lr = HyperParameterSearch(
        "imagenet",
        "lr",
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        verbose=1,
    )

    imagenet_lr.training()

    imagenet_epoch = HyperParameterSearch(
        "imagenet",
        "epoch",
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        verbose=1,
    )

    imagenet_epoch.training()

    imagenet_batch = HyperParameterSearch(
        "imagenet",
        "batch_size",
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        verbose=1,
    )
    imagenet_batch.training()


