import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from npal.classifier.base_classifier import BaseClassifier


class ScikitLearnClassifier(BaseClassifier):
    def __init__(self, args, classes, remaining_classes, imbalance_factors):
        super().__init__(args, classes, remaining_classes, imbalance_factors)

        # Set model
        if self.classifier_type == "svm":
            self.model = SVC(gamma="scale", random_state=args.seed)
        elif self.classifier_type == "logistic":
            max_iter = 100 if len(remaining_classes) == 2 else 500  # need more iterations to converge in multiclass
            self.model = LogisticRegression(
                solver="lbfgs", multi_class="auto", max_iter=max_iter, random_state=args.seed
            )
        else:
            raise ValueError()

    def fit(self, X, y):
        if len(X.shape) > 2:  # flatten image
            X = np.reshape(X, (len(X), -1))
        # y is not one-hot here
        sample_weights = None
        if self.args.use_class_weights_for_fit:
            # NOTE: Assumes class weights are ordered by value (e.g. 0 before 1 before 2, etc.)
            sample_weights = [self.class_weights[sorted(set(y)).index(label)] for label in y]

        self.model.fit(X, y, sample_weight=sample_weights)

    def predict(self, X):
        if len(X.shape) > 2:  # flatten image
            X = np.reshape(X, (len(X), -1))
        # Should return single value for prediction (not one-hot)
        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X):
        if len(X.shape) > 2:  # flatten image
            X = np.reshape(X, (len(X), -1))
        # Returns for all classes remaining in the dataset
        probs = self.model.predict_proba(X)
        return probs

    def get_num_classifier_features(self):
        if isinstance(self.model, SVC):
            # features = 1 f-score if binary, else num_classes f-scores.
            num_extra_features = 1 if len(self.remaining_classes) == 2 else len(self.remaining_classes)
        else:  # Logistic, CNN, ResNet
            # features = num_classes probabilities
            num_extra_features = len(self.remaining_classes)
        return num_extra_features

    def get_classifier_features(self, X):
        if len(X.shape) > 2:  # flatten image
            X = np.reshape(X, (len(X), -1))
        if isinstance(self.model, SVC):
            # (n_samples x n_classes) if multi one-vs-rest, (n_samples, ) if binary
            f_scores = self.model.decision_function(X)
            if len(f_scores.shape) == 1:  # binary
                f_scores = f_scores[:, None]
            return f_scores
        else:  # Logistic, CNN, ResNet
            return self.predict_proba(X)  # (n_samples x n_classes)

    def decision_function(self, X):
        if not isinstance(self.model, SVC):
            raise NotImplementedError("Decision function only implemented for SVM classifier.")
        if len(X.shape) > 2:  # flatten image
            X = np.reshape(X, (len(X), -1))
        return self.model.decision_function(X)

    def get_classifier_embedding(self, X):
        raise RuntimeError("ScikitLearnClassifier does not support model embeddings.")
