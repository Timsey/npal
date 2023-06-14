import pathlib
import numpy as np
from torchvision.datasets import CIFAR10


def load_cifar10_data(args):
    """
    32x32, 10 classes. Equal class distribution.
    50,000 training, 10,000 test
    """

    path = pathlib.Path(args.abs_path_to_data_store) / "CIFAR10"
    path.mkdir(parents=True, exist_ok=True)
    # data are numpy arrays, targets are lists
    training = CIFAR10(path, train=True, download=True)
    train_X, train_y = training.data, training.targets
    testing = CIFAR10(path, train=False, download=True)
    test_X, test_y = testing.data, testing.targets

    # Data is (N, H, W, C), so transform to (N, C, H, W)
    train_X = np.transpose(train_X, axes=(0, 3, 1, 2))
    test_X = np.transpose(test_X, axes=(0, 3, 1, 2))

    return train_X, np.array(train_y), test_X, np.array(test_y)
