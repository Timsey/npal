import pathlib
from torchvision.datasets import FashionMNIST


def load_fashion_mnist_data(args):
    """
    28x28, 10 classes. Equal class distribution.
    60,000 training, 10,000 test
    """

    path = pathlib.Path(args.abs_path_to_data_store) / "FashionMNIST"
    path.mkdir(parents=True, exist_ok=True)
    # All are Tensors
    training = FashionMNIST(path, train=True, download=True)
    train_X, train_y = training.data, training.targets
    testing = FashionMNIST(path, train=False, download=True)
    test_X, test_y = testing.data, testing.targets

    # (N, H, W) to (N, C=1, H, W)
    train_X = train_X.unsqueeze(1)
    test_X = test_X.unsqueeze(1)

    # To numpy
    return train_X.numpy(), train_y.numpy(), test_X.numpy(), test_y.numpy()
