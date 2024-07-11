from hyperparameter_search import HyperParameterSearch
from utility.dataloader import DataLoader

if __name__ == "__main__":
    iamgenet_data = DataLoader("imagenet")
    imagenet_train_ds, imagenet_val_ds, imagenet_test_ds = iamgenet_data.load_dataset()
    image_net_batch = HyperParameterSearch(
        "imagenet",
        hyperparameter="batch_size",
        train_ds=imagenet_train_ds,
        val_ds=imagenet_val_ds,
        test_ds=imagenet_test_ds,
        verbose=1,
    )
    image_net_batch.training()

    image_net_lr = HyperParameterSearch(
        "imagenet",
        hyperparameter="lr",
        train_ds=imagenet_train_ds,
        val_ds=imagenet_val_ds,
        test_ds=imagenet_test_ds,
        verbose=1,
    )
    image_net_lr.training()

    image_net_momentum = HyperParameterSearch(
        "imagenet",
        hyperparameter="momentum",
        train_ds=imagenet_train_ds,
        val_ds=imagenet_val_ds,
        test_ds=imagenet_test_ds,
        verbose=1,
    )
    image_net_momentum.training()
