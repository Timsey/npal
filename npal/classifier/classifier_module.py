import time

from npal.classifier.sklearn_classifiers import ScikitLearnClassifier
from npal.classifier.cnn_classifier import CNNClassifier
from npal.classifier.resnet_classifier import ResNetClassifier


class ALClassifier:
    def __init__(self, args, classes, remaining_classes, imbalance_factors, suppress_train_log=False):
        self.args = args
        self.classes = classes
        self.remaining_classes = remaining_classes
        self.imbalance_factors = imbalance_factors
        self.classifier_type = args.classifier_type
        self.score_type = args.score_type
        self.suppress_train_log = suppress_train_log

        zero = False
        for factor in imbalance_factors:
            if factor == 0:
                zero = True
            else:
                if zero:
                    # Need this check, because make_one_hot assumes that predict() classes are equal to the total
                    #  number of classes, but this is not true for sklearn classifiers specifically when the
                    #  imbalance factor for a particular class is 0: sklearn internally checks the number of classes
                    #  in the data, and its output will have predictions for the num_remaining_classes only. This is
                    #  good practice when writing classifiers, because we don't want to predict classes that don't
                    #  exist in the data, unless we explicitly want to model out-of-distribution classes.
                    # As a result, if we have - say - 10 classes, and set an imbalance factor of 0 for class - say -
                    #  5, then the classifier will assign class label 5 to class 6 instead (or actually 7, because
                    #  0-indexing, but whatever). This means our accuracy computations for each class beyond class 5
                    #  will be shifted one class to the lower end, which results in confusing evaluation.
                    # We could solve this by checking the imbalance factor during transformation to one-hot, but
                    #  we're not going to implement this until it becomes necessary.
                    # Note that imbalance factor 0 is fine to use for the last classes in numerical order, because
                    #  then class labels don't shift between classifier and one-hot representation.
                    raise RuntimeError(
                        "Using imbalance factor of 0 for any but the last classes (in numerical order) "
                        "is not supported. E.g. [1, 3, 5, 0, 0] is okay, but [1, 0, 3] is not."
                    )

        if self.classifier_type == "cnn":
            self.classifier = CNNClassifier(
                args, classes, remaining_classes, imbalance_factors, suppress_train_log=suppress_train_log
            )
        elif self.classifier_type == "resnet":
            self.classifier = ResNetClassifier(
                args, classes, remaining_classes, imbalance_factors, suppress_train_log=suppress_train_log
            )
        elif self.classifier_type in ["svm", "logistic"]:
            self.classifier = ScikitLearnClassifier(args, classes, remaining_classes, imbalance_factors)
        else:
            raise ValueError(f"Unknown classifier type {self.classifier_type}.")

    def make_reinitialised_classifier(self, suppress_train_log=False):
        # This returns an untrained instance of this classifier module, while keeping the original intact. Useful when
        # we want to train multiple classifiers as part of an inner loop.
        classifier_class = self.__class__
        return classifier_class(
            self.args,
            self.classes,
            self.remaining_classes,
            self.imbalance_factors,
            suppress_train_log=suppress_train_log,
        )

    def fit(self, X, y):
        # y is not one-hot here
        start_time = time.perf_counter()
        self.classifier.fit(X, y)
        extra_outputs = {"train_time": time.perf_counter() - start_time}
        return extra_outputs

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        probs = self.classifier.predict_proba(X)
        return probs

    def score(self, X, y, reduce="mean"):
        if self.score_type == "accuracy":
            return self.accuracy(X, y, reduce=reduce)
        else:
            raise NotImplementedError()

    def accuracy(self, X, y, reduce="mean"):
        # y is not one-hot here
        return self.classifier.accuracy(X, y, reduce=reduce)

    def binary_prec_rec(self, X, y):
        return self.classifier.binary_prec_rec(X, y)

    def get_classifier_features_for_np(self, X):
        return self.classifier.get_classifier_features(X)
