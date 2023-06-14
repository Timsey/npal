import logging
from npal.acquisition.heuristics import (
    RandomStrategy,
    MyopicOracleStrategy,
    FScoreStrategy,
    HALStrategy,
    KCenterGreedyStrategy,
    CBalEntropyGreedyStrategy,
    UncertaintySamplingStrategy,
)

from npal.acquisition.np_acquisition import NPStrategy
from npal.acquisition.gcn_acquisition import GCNStrategy


class ALStrategy:
    def __init__(self, args, data_module, classifier_module):
        self.args = args
        self.data_module = data_module
        self.classifier_module = classifier_module

        self.budget = args.budget

        self.logger = logging.getLogger("ALStrategy")

        if args.acquisition_strategy == "random":  # Random: default ML approach
            self.acquirer = RandomStrategy(args)
        elif args.acquisition_strategy == "oracle":  # Myopic oracle
            self.acquirer = MyopicOracleStrategy(args)
        elif args.acquisition_strategy == "fscore":
            if args.classifier_type != "svm":
                raise RuntimeError(
                    f"Can only use f-score for SVM classifier, but found classifier type: {args.classifier_type}."
                )
            self.acquirer = FScoreStrategy(args)
        elif args.acquisition_strategy == "kcgreedy":
            self.acquirer = KCenterGreedyStrategy(args)
        elif args.acquisition_strategy == "cbal":
            self.acquirer = CBalEntropyGreedyStrategy(args)
        elif "uc_" in args.acquisition_strategy:
            self.acquirer = UncertaintySamplingStrategy(args, mode=args.acquisition_strategy)
        elif "hal_" in args.acquisition_strategy:
            self.acquirer = HALStrategy(args, mode=args.acquisition_strategy)
        elif "np_" in args.acquisition_strategy:
            self.acquirer = NPStrategy(args, data_module, classifier_module, mode=args.acquisition_strategy)
        elif "gcn_" in args.acquisition_strategy:
            self.acquirer = GCNStrategy(args, data_module, classifier_module, mode=args.acquisition_strategy)
        else:
            raise ValueError(f"Unknown acquisition strategy: {args.acquisition_strategy}.")

    def propose_acquisition(self, classifier_module, **kwargs):
        return self.acquirer.propose_acquisition(self.data_module, classifier_module, **kwargs)

    def acquire_labels(self, classifier_module, **kwargs):
        # Acquire suggestions from strategy
        pool_inds_to_annotate, extra_outputs = self.propose_acquisition(classifier_module, **kwargs)

        # This mutates annot_X, annot_y, pool_X, and pool_y.
        self.data_module.annotate(pool_inds_to_annotate)
        annot_X, annot_y = self.data_module.get_annot_data().get_data()
        # Logging
        extra_outputs["current_labels"] = {cl: (cl == annot_y).sum() for cl in set(annot_y)}
        self.logger.info("Current labels: {}".format(extra_outputs["current_labels"]))

        return pool_inds_to_annotate, extra_outputs
