import random
import logging
import numpy as np

from sklearn.model_selection import train_test_split

from npal.data.base_data_loading import BaseDataLoader


class ImbalancedDataLoader(BaseDataLoader):
    def __init__(self, args):
        super().__init__(args)
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
        self.num_annot_points = args.num_annot_points
        self.num_pool_points = args.num_pool_points

        self.test_fraction = args.test_fraction
        self.val_fraction = args.val_fraction
        self.reward_fraction = args.reward_fraction
        self.pool_fraction = args.pool_fraction

        self.logger = logging.getLogger("ImbalancedDataLoader")

        if args.total_num_points is not None:
            assert (
                args.num_train_points is None
            ), "Cannot specify `num_train_points` and `total_num_points` simultaneously."
            assert args.num_val_points is None, "Cannot specify `num_val_points` and `total_num_points` simultaneously."
            assert (
                args.num_test_points is None
            ), "Cannot specify `num_test_points` and `total_num_points` simultaneously."

        self.imbalancer = Imbalancer(args)

    def __call__(self):
        train_X, train_y, test_X, test_y = self.load_data()  # Implemented by BaseDataLoader
        train_X, train_y, test_X, test_y = self.preprocess_data(train_X, train_y, test_X, test_y)
        train_X, train_y, val_X, val_y, reward_X, reward_y = self.split_train_val_reward(train_X, train_y)
        annot_X, annot_y, pool_X, pool_y = self.split_annot_pool(train_X, train_y)

        num_val_points = len(val_y) if val_y is not None else None
        num_reward_points = len(reward_y) if reward_y is not None else None
        self.logger.info(f"Data shape: {(1, *train_X.shape[1:])}")
        self.logger.info(f"{len(annot_y)} annotated, {len(pool_y)} pool")
        self.logger.info(
            f"{len(train_y)} train, {num_val_points} validation, " f"{len(test_y)} test, {num_reward_points} reward."
        )
        # Logging num datapoints per class
        self.logger.info("Train per class: {}".format(self.get_num_per_class(train_y)))
        self.logger.info("Annot per class: {}".format(self.get_num_per_class(annot_y)))
        self.logger.info("Pool per class: {}".format(self.get_num_per_class(pool_y)))
        return annot_X, annot_y, pool_X, pool_y, train_X, train_y, val_X, val_y, reward_X, reward_y, test_X, test_y

    def preprocess_data(self, train_X, train_y, test_X=None, test_y=None):
        if test_X is None:  # No test data provided yet
            assert test_y is None
            train_X, train_y = self.imbalance_data(train_X, train_y)
            train_X, _, _ = self.normalise_data(train_X)
            train_X, train_y, test_X, test_y = self.split_train_test(train_X, train_y)
        else:  # Train and test data provided
            assert set(train_y) == set(test_y), "Train or test data is missing classes!"
            # First imbalance data, then compute normalisation, to simulate actually only having access to imbalanced
            #  data.
            train_X, train_y = self.imbalance_data(train_X, train_y)
            test_X, test_y = self.imbalance_data(test_X, test_y)
            train_X, mean, std = self.normalise_data(train_X)
            test_X, _, _ = self.normalise_data(test_X, mean, std)
            # Further reduce dataset if we specified some absolute number of points to keep
            train_X, train_y, test_X, test_y = self.split_train_test(train_X, train_y, test_X, test_y)
        return train_X, train_y, test_X, test_y

    def imbalance_data(self, X, y):
        # Imbalance data here, to preserve class ratios between train and test.
        X, y = self.imbalancer(X, y)
        return X, y

    def split_train_test(self, X, y, test_X=None, test_y=None):
        # If test data is given then we don't have to actually split, but we do need to check that we keep the
        #  appropriate number of points if `total_num_points` or `num_test_points` is set.
        if test_X is not None:
            assert test_y is not None
            if self.total_num_points is not None:
                if self.total_num_points > len(y) + len(test_y):
                    raise RuntimeError(
                        f"Requested {self.total_num_points} data points, but data set only contains "
                        f"{len(y) + len(test_y)}; possibly this happens due to imbalancing."
                    )
                num_test_points = int(self.total_num_points * self.test_fraction)
                assert num_test_points <= len(
                    test_y
                ), "Too many test points requested. Reduce `test_fraction` or `total_num_points`."
                num_train_points = self.total_num_points - num_test_points
                assert num_train_points > 0, "Cannot have 0 train points."
                assert num_train_points <= len(
                    y
                ), "Too many train points requested. Increase `test_fraction` or reduce `total_num_points`."

                # Split off num_train_points from provided train data
                labels = y if self.stratify else None
                train_X, _, train_y, _ = train_test_split(
                    X,
                    y,
                    train_size=num_train_points,
                    test_size=None,
                    random_state=self.data_split_seed,
                    stratify=labels,
                )
                # Split off num_test_points from provided test data
                labels = test_y if self.stratify else None
                _, test_X, _, test_y = train_test_split(
                    test_X,
                    test_y,
                    train_size=None,
                    test_size=num_test_points,
                    random_state=self.data_split_seed,
                    stratify=labels,
                )
            elif self.num_test_points is not None:
                # `total_num_points` not set, so just split off `num_test_points if given`. Train is split off later.
                # Split off num_test_points from provided test data
                labels = test_y if self.stratify else None
                _, test_X, _, test_y = train_test_split(
                    test_X,
                    test_y,
                    train_size=None,
                    test_size=self.num_test_points,
                    random_state=self.data_split_seed,
                    stratify=labels,
                )
                train_X, train_y = X, y
            else:
                # No total number of points set and no number of test points set, so just continue with all data.
                train_X, train_y = X, y
        else:
            # Keep args.total_num_points only if set
            if self.total_num_points is not None:
                if self.total_num_points > len(y):
                    raise RuntimeError(
                        f"Requested {self.total_num_points} data points, but data set only contains "
                        f"{len(y)} after imbalancing."
                    )
                labels = y if self.stratify else None
                X, _, y, _ = train_test_split(
                    X,
                    y,
                    train_size=self.total_num_points,
                    test_size=None,
                    random_state=self.data_split_seed,
                    stratify=labels,
                )

                # Now split train, test using fractions
                labels = y if self.stratify else None
                train_X, test_X, train_y, test_y = train_test_split(
                    X,
                    y,
                    train_size=None,
                    test_size=self.test_fraction,
                    random_state=self.data_split_seed,
                    stratify=labels,
                )
            else:  # `total_num_points` not set, check if `num_test_points` is set; train is split off later.
                labels = y if self.stratify else None
                if self.num_test_points is not None:
                    train_X, test_X, train_y, test_y = train_test_split(
                        X,
                        y,
                        train_size=None,
                        test_size=self.num_test_points,
                        random_state=self.data_split_seed,
                        stratify=labels,
                    )
                else:  # `num_test_points` not set, so split using `test_fraction`
                    train_X, test_X, train_y, test_y = train_test_split(
                        X,
                        y,
                        train_size=None,
                        test_size=self.test_fraction,
                        random_state=self.data_split_seed,
                        stratify=labels,
                    )

        return train_X, train_y, test_X, test_y

    def split_train_val_reward(self, train_X, train_y):
        val_X, val_y = None, None
        labels = train_y if self.stratify else None
        if self.num_val_points is not None:
            if self.num_val_points > 0:  # Else keep train as is and val as None.
                # Split rest into train, val
                train_X, val_X, train_y, val_y = train_test_split(
                    train_X,
                    train_y,
                    train_size=None,
                    test_size=self.num_val_points,
                    random_state=self.data_split_seed,
                    stratify=labels,
                )
        else:
            if self.val_fraction > 0:
                # Split rest into train, val
                train_X, val_X, train_y, val_y = train_test_split(
                    train_X,
                    train_y,
                    train_size=None,
                    test_size=self.val_fraction,
                    random_state=self.data_split_seed,
                    stratify=labels,
                )

        # Split rest into train, reward
        reward_X, reward_y = None, None
        labels = train_y if self.stratify else None
        if self.num_reward_points is not None:
            if self.num_reward_points > 0:  # Else keep train as is and reward as None.
                train_X, reward_X, train_y, reward_y = train_test_split(
                    train_X,
                    train_y,
                    train_size=None,
                    test_size=self.num_reward_points,
                    random_state=self.data_split_seed,
                    stratify=labels,
                )
        else:
            reward_fraction = self.reward_fraction / (1 - self.val_fraction)
            if reward_fraction > 0:
                train_X, reward_X, train_y, reward_y = train_test_split(
                    train_X,
                    train_y,
                    train_size=None,
                    test_size=reward_fraction,
                    random_state=self.data_split_seed,
                    stratify=labels,
                )

        # Finally, split of num_train_points if set
        if self.num_train_points is not None:
            assert self.num_train_points > 0, "Cannot use 0 training points."
            labels = train_y if self.stratify else None
            train_X, _, train_y, _ = train_test_split(
                train_X,
                train_y,
                train_size=self.num_train_points,
                test_size=None,
                random_state=self.data_split_seed,
                stratify=labels,
            )

        return train_X, train_y, val_X, val_y, reward_X, reward_y

    def split_annot_pool(self, train_X, train_y):
        # Labeled-unlabeled (annotated-pool) split
        labels = train_y if self.stratify else None
        if self.num_annot_points:  # Set num_annot_points, rest is pool.
            annot_X, pool_X, annot_y, pool_y = train_test_split(
                train_X,
                train_y,
                train_size=self.num_annot_points,
                test_size=None,
                random_state=self.data_split_seed,
                stratify=labels,
            )
            if self.num_pool_points:  # Additionally throw away all points > num_pool_points from pool.
                labels = pool_y if self.stratify else None
                pool_X, _, pool_y, _ = train_test_split(
                    pool_X,
                    pool_y,
                    train_size=self.num_pool_points,
                    test_size=None,
                    random_state=self.data_split_seed,
                    stratify=labels,
                )
        elif self.num_pool_points:  # Set num_pool_points, rest is annot
            annot_X, pool_X, annot_y, pool_y = train_test_split(
                train_X,
                train_y,
                train_size=None,
                test_size=self.num_pool_points,
                random_state=self.data_split_seed,
                stratify=labels,
            )
        else:  # Use pool fraction
            annot_X, pool_X, annot_y, pool_y = train_test_split(
                train_X,
                train_y,
                train_size=None,
                test_size=self.pool_fraction,
                random_state=self.data_split_seed,
                stratify=labels,
            )
        return annot_X, annot_y, pool_X, pool_y

    @staticmethod
    def get_num_per_class(labels):
        return {cl: (labels == cl).sum() for cl in set(labels)}


class Imbalancer:
    def __init__(self, args):
        self.args = args
        self.classes = None
        self.remaining_classes = None
        self.imbalance_factors = np.array(args.imbalance_factors)
        self.ignore_existing_imbalance = args.ignore_existing_imbalance

    def __call__(self, X, y):
        assert len(y.shape) == 1, f"Incorrect label shape: {y.shape}. E.g., labels should not be one-hot here."

        if self.classes and self.classes != set(y):
            raise RuntimeError(
                "Imbalancer called multiple times, but with a different set of target classes! This will lead to "
                "inconsistencies in the number of classes in `imbalance_factors`."
            )
        # NOTE: If we want to use a different set of classes for different data partitions, then we still require
        #  that the below `self.classes` contains all classes in the entire dataset (else score calculations or
        #  training might fail).
        self.classes = set(y)

        if len(self.imbalance_factors) != len(self.classes):
            if len(self.imbalance_factors) == 1 and self.imbalance_factors[0] == 1.0:
                self.imbalance_factors = np.ones(len(self.classes))  # balanced
            else:
                raise RuntimeError(
                    f"`imbalance_factors` should be of length {len(self.classes)}, not {len(self.imbalance_factors)}."
                )

        imbalance_factor_max = np.max(self.imbalance_factors)
        # Imbalance the balanced training data according to imba_fracs.
        # Create dict mapping classes to inds
        indices_per_class = []
        # NOTE: This assumes that the lowest class label corresponds to the first entry of self.imbalance_factors.
        # This is consistent with the way scikit-learn indexes class labels.
        for c in sorted(self.classes):
            indices_per_class.append(np.where(y == c)[0].tolist())

        if self.ignore_existing_imbalance:
            # In case we are ignoring potential existing imbalance we just apply the
            # imbalance factor on top of whatever imbalance is already in the data.
            num_to_keep_per_class = [
                int(len(inds) * self.imbalance_factors[i] / imbalance_factor_max)
                for i, inds in enumerate(indices_per_class)
            ]
        else:
            # Not ignoring existing imbalance: make sure that eventual imbalance matches imbalance_factors, despite
            # existing data imbalance. This may reduce the number of data points available.
            max_num_inds_per_class = np.max([len(inds) for inds in indices_per_class])
            # Ideally, we would use this many points
            num_points_to_try = max_num_inds_per_class * self.imbalance_factors / imbalance_factor_max
            # However, if there is existing imbalance we might not be able to select this number for each class.
            # Determine the fraction we need to reduce this by to make sure every class can be sampled appropriately.
            # zero_fixed_num_points_to_try will be 0 where imbalance_factors is 0.
            nonzero_inds = num_points_to_try != 0
            zero_fixed_num_points_to_try = np.ones_like(num_points_to_try)
            zero_fixed_num_points_to_try[nonzero_inds] = num_points_to_try[nonzero_inds]
            subsample_fractions = np.array([len(inds) for inds in indices_per_class]) / zero_fixed_num_points_to_try
            # Subsample by this fraction
            num_to_keep_per_class = (np.min(subsample_fractions) * num_points_to_try).astype(int).tolist()

        # Get indices of points to keep and remove
        inds = []
        # NOTE: This is the same class order as used above.
        for c in sorted(self.classes):
            # Shuffle not in place
            # NOTE: This changes data order using the python random state.
            # state = random.getstate()
            # random.seed(self.args.data_split_seed)
            class_indices = random.sample(indices_per_class[c], len(indices_per_class[c]))
            # random.setstate(state)

            # Number of indices to select
            class_imba = class_indices[: num_to_keep_per_class[c]]
            inds.extend(class_imba)

        inds = np.array(inds)
        imbalanced_X = X[inds, ...]
        imbalanced_y = y[inds]
        if self.remaining_classes is not None:
            if self.remaining_classes != set(imbalanced_y):
                raise RuntimeError(
                    "`remaining_classes` changed from previous Imbalancer call. Probably called once on train and once "
                    "on test data, and test does not contain the same classes as train?"
                )
        else:
            self.remaining_classes = set(imbalanced_y)
        return imbalanced_X, imbalanced_y
