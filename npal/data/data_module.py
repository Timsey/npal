import functools
import numpy as np

from npal.data.imba_data_loading import ImbalancedDataLoader


class ALData:
    def __init__(self, args):
        self.args = args
        self.data_wrap = functools.partial(DataWrapper, args=args)

        self.data_loader = ImbalancedDataLoader(args)
        data = self.data_loader()  # call data loader
        # These properties are mutated by the data imbalancer when calling self.data_loader()
        self.classes = self.data_loader.imbalancer.classes
        self.remaining_classes = self.data_loader.imbalancer.remaining_classes
        self.imbalance_factors = self.data_loader.imbalancer.imbalance_factors

        annot_X, annot_y, pool_X, pool_y, train_X, train_y, val_X, val_y, reward_X, reward_y, test_X, test_y = data

        self.annot_X, self.annot_y = annot_X, annot_y
        self.pool_X, self.pool_y = pool_X, pool_y
        self.train_X, self.train_y = train_X, train_y
        self.val_X, self.val_y = val_X, val_y
        self.reward_X, self.reward_y = reward_X, reward_y
        self.test_X, self.test_y = test_X, test_y

    def get_annot_data(self):
        return self.data_wrap(self.annot_X, self.annot_y)

    def get_pool_data(self):
        return self.data_wrap(self.pool_X, self.pool_y)

    def get_train_data(self):
        return self.data_wrap(self.train_X, self.train_y)

    def get_val_data(self):
        if self.val_X is None or self.val_y is None:
            raise RuntimeError("Validation data was not split off.")
        return self.data_wrap(self.val_X, self.val_y)

    def get_test_data(self):
        if self.test_X is None or self.test_y is None:
            raise RuntimeError("Test data was not split off.")
        return self.data_wrap(self.test_X, self.test_y)

    def get_reward_data(self):
        if self.reward_X is None or self.reward_y is None:
            raise RuntimeError("Reward data was not split off.")
        return self.data_wrap(self.reward_X, self.reward_y)

    def annotate(self, pool_inds_to_annotate):
        pool_inds_to_annotate = np.array(pool_inds_to_annotate)
        X_to_annotate = self.pool_X[pool_inds_to_annotate, ...]
        y_to_annotate = self.pool_y[pool_inds_to_annotate]
        # These are numpy arrays
        self.annot_X = np.append(self.annot_X, X_to_annotate, axis=0)
        self.annot_y = np.append(self.annot_y, y_to_annotate, axis=0)

        # Remove from pool: do this by first calculating which indices should still
        #  be in the pool, and then just slicing.
        all_pool_inds = set(range(len(self.pool_y)))
        # XOR between all pool indices and the indices to move gives remaining indices.
        pool_inds_to_keep = np.array(list(all_pool_inds ^ set(pool_inds_to_annotate)))
        self.pool_X = self.pool_X[pool_inds_to_keep, ...]
        self.pool_y = self.pool_y[pool_inds_to_keep]

    def __repr__(self):
        num_annot = len(self.get_annot_data())
        num_pool = len(self.get_pool_data())
        # NOTE: `num_train` is not equal to len(self.get_train_data()) in the case that we have set
        #  both num_annot_point and num_pool_points as run arguments.
        num_train = num_annot + num_pool
        try:
            num_val = len(self.get_val_data())
        except RuntimeError:
            num_val = 0
        try:
            num_test = len(self.get_test_data())
        except RuntimeError:
            num_test = 0
        try:
            num_reward = len(self.get_reward_data())
        except RuntimeError:
            num_reward = 0

        descr = "Annot + Pool = Train: {} + {} = {}. Val: {}, Test: {}, Reward: {}.".format(
            num_annot, num_pool, num_train, num_val, num_test, num_reward
        )
        return descr


class DataWrapper:
    def __init__(self, X, y, args=None, flatten=False):
        self.args = args
        self.flatten = flatten
        self.X = X
        self.y = y
        assert len(X) == len(y), "X and y don't contain equal number of points!"

    def get_data(self):
        if self.flatten:
            return self.X.reshape(len(self.y), -1), self.y
        else:
            return self.X, self.y

    def get_max_of_abs(self):
        # Return maximum absolute value in X
        X, _ = self.get_data()
        return np.max(np.abs(X))

    def __len__(self):
        return len(self.y)
