import abc
import numpy as np

from sklearn.metrics import precision_score, recall_score


class BaseClassifier(abc.ABC):
    def __init__(self, args, classes, remaining_classes, imbalance_factors):
        self.args = args
        # Used for one-hot vector representation for accuracy computations (original number of classes, but the
        #  actual classifier may use fewer classes because classes were removed due to imbalancing).
        self.classes = classes
        # Used internally by classifiers
        self.remaining_classes = remaining_classes
        self.imbalance_factors = imbalance_factors

        self.classifier_type = args.classifier_type

        self.class_weights = np.zeros_like(imbalance_factors)
        nonzero_classes = imbalance_factors != 0
        imbalance_factors_to_norm = imbalance_factors[nonzero_classes]
        if self.args.class_weight_type is None:
            # use weight 1 for all classes
            class_weights = np.ones_like(imbalance_factors_to_norm)
        elif "-" in self.args.class_weight_type:
            # Use directly specified class weights
            class_weights = list(map(float, self.args.class_weight_type.split("-")))
            # Normalise these class weights
            class_weights /= np.sum(class_weights)
            if len(class_weights) != len(self.classes):
                raise ValueError(
                    f"Number of specified class weights ({len(class_weights)}) does not match "
                    f"number of remaining classes in dataset ({len(nonzero_classes)})!"
                )
        elif self.args.class_weight_type == "balanced":
            # The rare class has lower imbalance_factor, so should get higher weight.
            # use weighting where the average weight is 1
            class_weights = 1 / (imbalance_factors_to_norm * np.mean(1 / imbalance_factors_to_norm))
        elif self.args.class_weight_type == "inverse":
            # Weighted quadratically, so that rare class ends up being more important by exactly the imbalance factor.
            # use weighting where the average weight is 1
            class_weights = 1 / (imbalance_factors_to_norm ** 2 * np.mean(1 / imbalance_factors_to_norm ** 2))
        else:
            raise ValueError(f"Unknown value for `class_weights`: {self.args.class_weight_type}.")

        self.class_weights[nonzero_classes] = class_weights

        # Set this in subclass
        self.model = None

    def fit(self, X, y):
        # y is not one-hot here
        # Implement this class (returns nothing, but trains self.model
        pass

    def predict(self, X):
        # Should return single value for prediction (not one-hot)
        pass

    def predict_proba(self, X):
        # Returns probs for all classes remaining in the dataset (self.num_remaining_classes)
        pass

    def get_num_classifier_features(self):
        # Should return the number of classifier features that can be used as extra input to the learned AL models.
        pass

    def get_classifier_features(self, X):
        # This method should return features specific to the classifier, to be used as extra input to the learned
        #  AL strategies. The number of features should match whatever is returned by get_num_classifier_features().
        pass

    def accuracy(self, X, y, reduce="mean"):
        # y is not one-hot here
        predictions = self.predict(X)
        predictions_one_hot = self.make_one_hot(predictions)
        y_one_hot = self.make_one_hot(y)

        # Get places where prediction is correct
        correct_inds = (y_one_hot == predictions_one_hot).sum(axis=1) == len(self.classes)
        # Use that information to get all the correct predictions (so we can check per class)
        correct_vals = np.where(correct_inds[:, None], y_one_hot, np.zeros_like(y_one_hot))
        # Sum over points to get number of correct predictions per class as array
        correct_per_class = correct_vals.sum(axis=0)
        # Total number of examples per class as array
        total_per_class = y_one_hot.sum(axis=0)
        for i, val in enumerate(total_per_class):
            # If there are none of a certain class, set the number of points for that class to 1 to avoid div 0 errors
            if val == 0.0:
                total_per_class[i] = 1

        # Note that since we compute per class this is equivalent to recall-per-class:
        #  TruePos / (TruePos + FalseNeg) = TruePos / NumClass
        # True accuracy is (TruePos + TrueNeg) / TotalPoints, which for binary classification is the same number
        #  for both classes: hence we do a per-class version.
        # This means the weighted accuracy here can also be viewed as an average recall.
        accs = correct_per_class / total_per_class

        # Class weighted mean accuracy
        # NOTE: If there are classes with 0 imbalance_factor, then self.class_weights will have mean < 1, but this
        #  does not matter for the accuracy computation, because those classes do not contribute to the accuracy; i.e.
        #  it is as if those classes did not exist, and self.class_weights mean of all the existing classes is 1, as
        #  required for accurate accuracy computation.
        reduced_accs = (correct_per_class * self.class_weights).sum() / (total_per_class * self.class_weights).sum()

        if reduce == "both":
            return reduced_accs, accs
        elif reduce == "mean":
            return reduced_accs
        elif reduce is None:
            return accs
        else:
            raise RuntimeError(f"Unknown `reduce` value: {reduce}.")

    def binary_prec_rec(self, X, y):
        """Binary precision-recall."""
        if len(set(y)) > 2:
            raise ValueError(
                "Can only compute binary precision-recall for binary classification, "
                "but found {} classes.".format(len(set(y)))
            )
        predictions = self.predict(X)
        tp0 = np.sum([a and b for a, b in zip(y == 0, 0 == predictions)])  # tn1
        tp1 = np.sum([a and b for a, b in zip(y == 1, 1 == predictions)])  # tn0
        fp0 = np.sum([a and b for a, b in zip(y == 1, 0 == predictions)])  # fn1
        fp1 = np.sum([a and b for a, b in zip(y == 0, 1 == predictions)])  # fn0
        # Duplicates of the above, but w.r.t. the opposite class.
        # tn0 = np.sum([a and b for a, b in zip(y == 1, 1 == predictions)])  # tp1
        # tn1 = np.sum([a and b for a, b in zip(y == 0, 0 == predictions)])  # tp0
        # fn0 = np.sum([a and b for a, b in zip(y == 0, 1 == predictions)])  # fp1
        # fn1 = np.sum([a and b for a, b in zip(y == 1, 0 == predictions)])  # fp0

        class0_prec = precision_score(y, predictions, pos_label=0)
        class1_prec = precision_score(y, predictions, pos_label=1)
        class0_rec = recall_score(y, predictions, pos_label=0)
        class1_rec = recall_score(y, predictions, pos_label=1)
        return class0_prec, class1_prec, class0_rec, class1_rec, {"tp0": tp0, "tp1": tp1, "fp0": fp0, "fp1": fp1}

    def make_one_hot(self, y):
        assert len(y.shape) == 1, f"y should be of shape (num_datapoint, ), not: {y.shape}."
        num_classes = len(self.classes)
        # Make sure one-hot works when label set is not simply (0, 1, ...), but for instance (7, 9)
        oy = np.zeros((y.size, num_classes))
        oy[np.arange(y.size), y] = 1
        return oy
