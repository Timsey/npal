import abc
import logging

from npal.data.datasets.uci import load_uci_data, UCI_DATA_NAMES
from npal.data.datasets.mnist import load_mnist_data
from npal.data.datasets.fashion_mnist import load_fashion_mnist_data
from npal.data.datasets.svhn import load_svhn_data
from npal.data.datasets.cifar10 import load_cifar10_data


class BaseDataLoader(abc.ABC):
    def __init__(self, args):
        self.args = args
        self.data_type = args.data_type

        self.stratify = args.stratify
        self.normalise_features = args.normalise_features
        self.data_split_seed = args.data_split_seed

        self.total_num_points = args.total_num_points
        self.num_test_points = args.num_test_points
        self.num_train_points = args.num_train_points
        self.num_val_points = args.num_val_points
        self.num_reward_points = args.num_reward_points

        self.test_fraction = args.test_fraction
        self.val_fraction = args.val_fraction
        self.reward_fraction = args.reward_fraction
        self.pool_fraction = args.pool_fraction

        self.logger = logging.getLogger("BaseDataLoader")

        if args.total_num_points is not None:
            assert (
                args.num_train_points is None
            ), "Cannot specify `num_train_points` and `total_num_points` simultaneously."
            assert args.num_val_points is None, "Cannot specify `num_val_points` and `total_num_points` simultaneously."
            assert (
                args.num_test_points is None
            ), "Cannot specify `num_test_points` and `total_num_points` simultaneously."

    def __call__(self):
        """
        Calling this method should load data from disk / simulate data and split the data into the proper subsets.

        See ImbalancedDataLoader in imba_data_loading for an example.

        returns: annot_X, annot_y, pool_X, pool_y, train_X, train_y, val_X, val_y, reward_X, reward_y, test_X, test_y
        """
        raise NotImplementedError()

    def load_data(self):
        """
        Loads data from disk / creates data by simulation. Note that Moons and UCI do not return test data, so
        this should be split of separately in self.__call__. See ImbalancedDataLoader in imba_data_loading for an
        example.
        """

        # Load train and test partitions.
        if self.data_type == "mnist":
            train_X, train_y, test_X, test_y = load_mnist_data(self.args)
        elif self.data_type == "fashion_mnist":
            train_X, train_y, test_X, test_y = load_fashion_mnist_data(self.args)
        elif self.data_type == "cifar10":
            train_X, train_y, test_X, test_y = load_cifar10_data(self.args)
        elif self.data_type == "svhn":
            train_X, train_y, test_X, test_y = load_svhn_data(self.args)
        elif self.data_type in UCI_DATA_NAMES:
            train_X, train_y, test_X, test_y = load_uci_data(self.args)
        else:
            raise ValueError(f"Unknown data_type: {self.data_type}.")

        # X is of shape, either: (N, F) for flat data, or (N, C, H, W) for images.
        # y is of shape (N, )
        return train_X, train_y, test_X, test_y

    def normalise_data(self, X, mean=None, std=None):
        """
        Normalise data.
        """

        if self.normalise_features:
            if self.data_type in UCI_DATA_NAMES:  # No separate test data is provided
                if mean is None or std is None:
                    mean = X.mean()
                    std = X.std()
                X = (X - mean) / std
            elif self.data_type in ["mnist", "fashion_mnist"]:
                if mean is None or std is None:
                    mean = X.mean()
                    std = X.std()
                X = (X - mean) / std
            elif self.data_type in ["svhn", "cifar10"]:
                # Multiple channels, so flatten per channel.
                if mean is None or std is None:
                    # Find stats of every channel separately
                    mean = X.mean(axis=(0, 2, 3)).reshape(1, 3, 1, 1)
                    std = X.std(axis=(0, 2, 3)).reshape(1, 3, 1, 1)
                X = (X - mean) / std
            else:
                raise ValueError("Unexpected `data_type`: {}.".format(self.data_type))
        else:
            mean = 1
            std = 1
        return X, mean, std
