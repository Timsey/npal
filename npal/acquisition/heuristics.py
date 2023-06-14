import copy
import time
import wandb
import random
import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.svm import SVC
from argparse import ArgumentParser
from sklearn.metrics import pairwise_distances

from npal.acquisition.acquisition_utils import compute_myopic_rewards
from npal.acquisition.base_acquisition import BaseStrategy


class RandomStrategy(BaseStrategy):
    """
    Random strategy: label points uniformly at random. This is the default ML setting.
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = "Random"

    def propose_acquisition(self, data_module, classifier_module, **kwargs):
        start_time = time.perf_counter()

        pool_inds_to_annotate = np.random.choice(
            range(len(data_module.get_pool_data())), size=self.budget, replace=False
        )

        extra_outputs = {
            "time": time.perf_counter() - start_time,
        }
        return pool_inds_to_annotate, extra_outputs


class MyopicOracleStrategy(BaseStrategy):
    """
    Myopic Oracle: select point that maximises classifier score after adding that point with true label and
    retraining the underlying classifier. If budget > 1, still select based only on myopic (1-step lookahead) score.
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = "MyopicOracle"
        self.eval_split = args.eval_split
        self.logger = logging.getLogger(self.name)
        self.report_interval = 100

    def propose_acquisition(self, data_module, classifier_module, **kwargs):
        if self.eval_split == "val":
            eval_data = data_module.get_val_data()
        elif self.eval_split == "test":
            eval_data = data_module.get_test_data()
        elif self.eval_split == "reward":
            eval_data = data_module.get_reward_data()
        else:
            raise ValueError(f"Unknown value for `eval_split`: {self.eval_split}.")

        annot_X, annot_y = data_module.get_annot_data().get_data()  # Automatically gets feature data instead of raw
        pool_X, pool_y = data_module.get_pool_data().get_data()  # Automatically gets feature data instead of raw

        # Compute improvements that happen on actual retraining
        improvements, time_taken = compute_myopic_rewards(
            classifier_module,
            eval_data,
            annot_X,
            annot_y,
            pool_X,
            pool_y,
            report_interval=self.report_interval,
            logger=self.logger,
        )

        extra_outputs = {
            "oracle_improvements": improvements,
            "time": time_taken,
        }
        if self.args.wandb:
            al_step = kwargs["al_step"]
            plt.bar(list(range(len(improvements))), improvements)
            plt.title("Improvements per action", fontsize=18)
            plt.ylabel("Improvement", fontsize=15)
            plt.xlabel("Action index", fontsize=15)
            wandb.log({f"oracle_improvements_step{al_step}": wandb.Image(plt)})
            plt.close()

        # NOTE: If there are multiple winners, then this chooses the winner based on the order in which
        #  the pool points are seen.
        pool_inds_to_annotate = np.argsort(improvements)[-self.budget :]

        return pool_inds_to_annotate, extra_outputs


class UncertaintySamplingStrategy(BaseStrategy):
    """
    Uncertainty sampling.

    Entropy, margin, least confidence.
    """

    def __init__(self, args, mode):
        super().__init__(args)
        self.mode = mode

        if mode == "uc_entropy":
            self.name = "UC-Entropy"
            self.method = "entropy"
        elif mode == "uc_margin":
            self.name = "UC-Margin"
            self.method = "margin"
        elif mode == "uc_lc":
            self.name = "UC-LeastConfidence"
            self.method = "least_conf"
        else:
            raise RuntimeError("Unknown acquisition mode: {}.".format(mode))

    def propose_acquisition(self, data_module, classifier_module, **kwargs):
        start_time = time.perf_counter()

        pool_X, _ = data_module.get_pool_data().get_data()
        # Assume probabilistic model
        probs = classifier_module.predict_proba(pool_X)

        if self.method == "entropy":
            entropy = np.sum(probs * np.log(probs + 1e-8), axis=1)  # Neg entropy
            pool_inds_to_annotate = np.argsort(entropy, axis=0)[: self.budget]  # Minimal neg entropy (max entropy)
            certainty_scores = entropy
        elif self.method == "margin":
            # Margin scores
            sorted_probs = np.sort(probs, axis=1)
            margins = np.abs(sorted_probs[:, -1] - sorted_probs[:, -2])
            pool_inds_to_annotate = np.argsort(margins, axis=0)[: self.budget]  # Minimal margin
            certainty_scores = margins
        elif self.method == "least_conf":
            max_probs = np.max(probs, axis=1)
            pool_inds_to_annotate = np.argsort(max_probs, axis=0)[: self.budget]  # Minimal max_prob
            certainty_scores = max_probs
        else:
            raise RuntimeError("Unknown acquisition mode: {}.".format(self.method))

        extra_outputs = {
            "certainty_scores": certainty_scores,
            "time": time.perf_counter() - start_time,
        }
        return np.array(pool_inds_to_annotate), extra_outputs


class FScoreStrategy(BaseStrategy):
    """
    SVM F-Score heuristic: select points closest to separating hyperplane (lowest absolute f-score).
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = "FScore"

    def propose_acquisition(self, data_module, classifier_module, **kwargs):
        if not isinstance(classifier_module.classifier.model, SVC):
            raise RuntimeError(
                "Can only use f-score for SVM classifier, but found classifier type: {}.".format(
                    classifier_module.classifier.model
                )
            )
        start_time = time.perf_counter()

        pool_X, _ = data_module.get_pool_data().get_data()
        f_values = classifier_module.classifier.decision_function(pool_X)
        if len(f_values.shape) == 2:  # Multiclass
            # Use the point that is closest to any decision boundary for now
            f_values = np.min(np.abs(f_values), axis=1)
        # Closest to hyperplane = smallest f_values
        pool_inds_to_annotate = np.argsort(np.abs(f_values))[: self.budget]

        extra_outputs = {
            "f_values": f_values,
            "time": time.perf_counter() - start_time,
        }
        return pool_inds_to_annotate, extra_outputs


class HALStrategy(BaseStrategy):
    """
    HAL heuristic implementation from "Active Learning for Skewed Data Sets" (2020).
    Paper link: https://arxiv.org/pdf/2005.11442v1.pdf

    NOTE: Original paper only used budget 1.
    """

    def __init__(self, args, mode):
        super().__init__(args)

        if mode == "hal_uni":
            self.name = "HAL-Uniform"
            self.method = "uniform"
        elif mode == "hal_gauss":
            self.name = "HAL-Gauss"
            self.method = "gauss"
        else:
            raise RuntimeError("Unknown acquisition mode: {}.".format(mode))

        self.hal_delta = args.hal_delta
        # Margin sampling if this is 1.
        # Random sampling if this is 0 and exploration is `uniform`.
        # Density sampling if this is 0 and exploration is `gaussian`.
        self.hal_exploit_p = args.hal_exploit_p

        self.logger = logging.getLogger(self.name)

    def propose_acquisition(self, data_module, classifier_module, **kwargs):
        start_time = time.perf_counter()

        pool_X, _ = data_module.get_pool_data().get_data()
        # Assume probabilistic model
        probs = classifier_module.predict_proba(pool_X)
        sorted_probs = np.sort(probs, axis=1)
        # Margin scores
        certainty_scores = np.abs(sorted_probs[:, -1] - sorted_probs[:, -2])

        if self.method == "uniform":
            exploration_scores = np.random.uniform(size=len(certainty_scores))
        elif self.method == "gauss":
            annot_X, _ = data_module.get_annot_data().get_data()
            # Flatten
            pool_X = pool_X.reshape(len(pool_X), -1)
            annot_X = annot_X.reshape(len(annot_X), -1)
            exploration_scores = 0
            for annot_point in annot_X:
                exploration_scores += np.exp(-1 * np.sum((pool_X - annot_point) ** 2, axis=1) / self.hal_delta)
        else:
            raise RuntimeError(f"Unknown HAL exploration type: {self.method}.")

        pool_inds_to_annotate, exploit_yes_no = [], []
        top_certainty = np.argsort(certainty_scores)
        top_exploration = np.argsort(exploration_scores)
        for _ in range(self.budget):
            # Note that this implementation may cause the same point to sampled twice, if it is both in the topk of
            #  certainty scores and in the topk of exploration scores.
            if random.uniform(0, 1) < self.hal_exploit_p:
                num_exploited = sum(exploit_yes_no)
                ind = top_certainty[-(num_exploited + 1)]
                pool_inds_to_annotate.append(ind)
                exploit_yes_no.append(True)
            else:
                num_explored = len(exploit_yes_no) - sum(exploit_yes_no)
                ind = top_exploration[-(num_explored + 1)]
                pool_inds_to_annotate.append(ind)
                exploit_yes_no.append(False)

        extra_outputs = {
            "certainty_scores": certainty_scores,
            "exploration_scores": exploration_scores,
            "exploit": exploit_yes_no,
            "time": time.perf_counter() - start_time,
        }
        if len(set(pool_inds_to_annotate)) < len(pool_inds_to_annotate):
            self.logger.warning(
                "Sampled duplicates: {}/{} unique.".format(len(set(pool_inds_to_annotate)), len(pool_inds_to_annotate))
            )
        return np.array(pool_inds_to_annotate), extra_outputs

    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # HAL heuristic args
        parser.add_argument(
            "--hal_exploit_p",
            default=0.5,
            type=float,
            help="Probability of exploitation (vs exploration) in the HAL heuristic.",
        )
        parser.add_argument(
            "--hal_delta",
            default=10,
            type=float,
            help="Delta to use for the HAL heuristic in the case of Gaussian exploration.",
        )

        return parser


class KCenterGreedyStrategy(BaseStrategy):
    """
    Returns point(s) that minimize(s) the maximum distance of any point to any center.

    From: "A Geometric Approach to Active Learning for Convolutional Neural Networks" (2017)
    Paper link: https://arxiv.org/abs/1708.
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = "KCenterGreedy"

    def propose_acquisition(self, data_module, classifier_module, **kwargs):
        start_time = time.perf_counter()

        annot_X, _ = data_module.get_annot_data().get_data()
        pool_X, _ = data_module.get_pool_data().get_data()
        # Flatten
        annot_X = annot_X.reshape(len(annot_X), -1)
        pool_X = pool_X.reshape(len(pool_X), -1)

        pool_inds_to_annotate = []
        min_distances_per_step = []
        centers = copy.deepcopy(annot_X)
        for _ in range(self.budget):
            # Default distance is euclidian / l2
            distances = pairwise_distances(pool_X, centers)  # NOTE: This is slow when pool_X is very large.
            # For every pool point, this gives the minimum distance to any cluster center (annotated points)
            min_distances = np.min(distances, axis=1)
            # Grab point that has the largest minimum distance: e.g. grab the point that is furthest from any center.
            # This process thus minimises the maximum distance from any center.
            pool_ind = np.argmax(min_distances)
            pool_inds_to_annotate.append(pool_ind)
            # Append selected point to centers for next round if budget > 1
            centers = np.append(centers, pool_X[pool_ind : pool_ind + 1, ...], axis=0)
            # For logging
            min_distances_per_step.append(min_distances)

        extra_outputs = {
            "min_distances_per_step": min_distances_per_step,
            "time": time.perf_counter() - start_time,
        }
        return np.array(pool_inds_to_annotate), extra_outputs


class CBalEntropyGreedyStrategy(BaseStrategy):
    """
    Returns point(s) that combine entropy sampling with a class-balancing regulariser.

    From: "Class-Balanced Active Learning for Image Classification" (2021)
    Paper link: https://arxiv.org/pdf/2110.04543v1.pdf.
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = "CBalEntropyGreedy"
        # NOTE: If this is 0 then we are simply doing entropy sampling (no regulariser)
        self.cbal_lambda = args.cbal_lambda

    def propose_acquisition(self, data_module, classifier_module, **kwargs):
        start_time = time.perf_counter()

        annot_X, annot_y = data_module.get_annot_data().get_data()
        pool_X, _ = data_module.get_pool_data().get_data()

        pool_inds_to_annotate = []
        cbal_scores_per_step = []
        num_labeled = len(annot_y)
        num_labeled_per_class = np.array([np.sum(annot_y == i) for i in sorted(set(annot_y))])
        # Assume probabilistic model
        probs = classifier_module.predict_proba(pool_X)
        for _ in range(self.budget):
            num_classes = probs.shape[1]
            # Compute entropies
            # shape = (num_pool,)
            neg_entropies = np.sum(probs * np.log(probs + 1e-8), axis=1)
            # NOTE: This assumes we are only using the remaining classes in our target distribution.
            zeros = np.zeros((num_classes, 1))
            num_to_label_per_class = (num_labeled / num_classes - num_labeled_per_class)[:, None]
            # shape = (num_pool x num_classes)
            target_distribution = np.max(np.concatenate((zeros, num_to_label_per_class), axis=1), axis=1)
            # Objective: shape = (num_pool,)
            # For high imbalance this essentially grabs the point with highest probability to belong to the rare class
            cbal_scores = neg_entropies + self.cbal_lambda * np.abs(target_distribution - probs).sum(axis=1)
            pool_ind = np.argmin(cbal_scores)
            pool_inds_to_annotate.append(pool_ind)
            # NOTE: Technically we should also be increasing num_labeled_per_class by 1 for the class of the labeled
            #  point, but we don't know the class yet here, so we don't do this. Probably then incrementing
            #  num_labeled here does very little as well, as it only slightly increases the regularisation
            #  term by 1 / num_classes.
            num_labeled += 1
            probs = np.delete(probs, pool_ind, axis=0)
            cbal_scores_per_step.append(cbal_scores_per_step)

        extra_outputs = {
            "cbal_scores_per_step": cbal_scores_per_step,
            "time": time.perf_counter() - start_time,
        }
        return np.array(pool_inds_to_annotate), extra_outputs

    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # CBAL heuristic args
        parser.add_argument(
            "--cbal_lambda",
            default=1,
            type=float,
            help="Regularisation coefficient lambda to use for the Class-Balancing AL heuristic.",
        )

        return parser
