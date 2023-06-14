import pathlib
from torchvision.datasets import MNIST


def load_mnist_data(args):
    """
    28x28, 10 classes.
    60,000 training, 10,000 test
    """

    path = pathlib.Path(args.abs_path_to_data_store) / "MNIST"
    path.mkdir(parents=True, exist_ok=True)
    # All are Tensors
    training = MNIST(path, train=True, download=True)
    train_X, train_y = training.data, training.targets
    testing = MNIST(path, train=False, download=True)
    test_X, test_y = testing.data, testing.targets

    # (N, H, W) to (N, C=1, H, W)
    train_X = train_X.unsqueeze(1)
    test_X = test_X.unsqueeze(1)

    # To numpy
    return train_X.numpy(), train_y.numpy(), test_X.numpy(), test_y.numpy()


# def load_mnist_data(args):
#     # MNIST from torch
#     train_X, train_y = torch.load(args.mnist_path / "training.pt")
#     test_X, test_y = torch.load(args.mnist_path / "test.pt")
#
#     # Flatten for e.g. SVM classifier
#     if args.image_flatten:
#         train_X = torch.flatten(train_X, start_dim=1, end_dim=-1)
#         test_X = torch.flatten(test_X, start_dim=1, end_dim=-1)
#
#     # To numpy
#     return train_X.numpy(), train_y.numpy(), test_X.numpy(), test_y.numpy()
