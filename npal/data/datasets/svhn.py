import pathlib
from torchvision.datasets import SVHN


def load_svhn_data(args):
    """
    32x32, 10 classes
    73,257 training, 26,032 test

    Potentially 531,131 extra (easier) samples by setting split="extra".
    """

    path = pathlib.Path(args.abs_path_to_data_store) / "SVHN"
    path.mkdir(parents=True, exist_ok=True)
    # All are numpy arrays
    training = SVHN(path, split="train", download=True)
    train_X, train_y = training.data, training.labels
    testing = SVHN(path, split="test", download=True)
    test_X, test_y = testing.data, testing.labels

    # Data is (N, C, H, W)
    return train_X, train_y, test_X, test_y
