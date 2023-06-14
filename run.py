import random
import logging
import torch
import wandb
import numpy as np

from datetime import datetime
from pprint import pformat

from npal.args import create_arg_parser
from npal.utils import log_hist_of_labels
from npal.wandb_utils import wandb_setup
from npal.data.data_module import ALData
from npal.classifier.classifier_module import ALClassifier
from npal.acquisition.acquisition_module import ALStrategy


def do_active_learning_run(args, logger):
    # Load and create train, val, test (, reward) data. Perform any necessary imbalancing transformations.
    data_module = ALData(args)
    logger.info("Data splits: {}".format(data_module))
    if args.eval_split == "val":
        eval_data = data_module.get_val_data()
    elif args.eval_split == "test":
        eval_data = data_module.get_test_data()
    elif args.eval_split == "reward":
        eval_data = data_module.get_reward_data()
    else:
        raise ValueError(f"`eval_split` should be 'val' or 'test', not {args.eval_split}.")

    eval_str = "task_{}_{}".format(args.eval_split, args.score_type)

    # Setup for AL step
    # Imbalance factors may be changed by data_module.imbalancer
    classifier_module = ALClassifier(
        args, data_module.classes, data_module.remaining_classes, data_module.imbalance_factors
    )
    # Active Learning strategy
    al_strategy = ALStrategy(args, data_module, classifier_module)
    logger.info(f"AL Strategy: {al_strategy.acquirer}")
    logger.info(f"AL Classifier: {classifier_module.classifier.model}")

    # Train classifier
    annot_X, annot_y = data_module.get_annot_data().get_data()
    if args.wandb:
        wandb.log({f"labels_{0}": [(cl == annot_y).sum() for cl in set(annot_y)]})
        log_hist_of_labels(annot_y, 0)

    extra_outputs = classifier_module.fit(annot_X, annot_y)
    # Evaluate classifier performance
    score, score_per_class = classifier_module.score(eval_data.X, eval_data.y, reduce="both")
    logger.info("Starting score: {:.3f}, per class: {}".format(score, ", ".join(f"{sc:.3f}" for sc in score_per_class)))

    if len(data_module.remaining_classes) == 2:  # Binary precision-recall
        s_class0_prec, s_class1_prec, s_class0_rec, s_class1_rec, pos_neg_dict = classifier_module.binary_prec_rec(
            eval_data.X, eval_data.y
        )
        logger.info(
            "Starting precision-recall. Class 0: ({:.3f}, {:.3f}); Class 1: ({:.3f}, {:.3f})".format(
                s_class0_prec, s_class0_rec, s_class1_prec, s_class1_rec
            )
        )
    logger.info("Classifier train time: {:.2f}s".format(extra_outputs["train_time"]))
    score_per_step, score_per_class_per_step = [score], [score_per_class]

    if args.wandb:
        wandb_dict = {eval_str + f"_class{cl}": sc for cl, sc in enumerate(score_per_class)}
        wandb_dict.update({"al_step": 0, eval_str: score})
        if len(data_module.remaining_classes) == 2:  # Binary precision-recall
            wandb_dict.update(
                {
                    "class0_prec": s_class0_prec,
                    "class0_rec": s_class0_rec,
                    "class1_prec": s_class1_prec,
                    "class1_rec": s_class1_rec,
                }
            )
            wandb_dict.update(pos_neg_dict)
        wandb.log(wandb_dict)

    assert args.al_steps > 0, f"Number of AL steps should be greater than 0, not: {args.al_steps}."
    annotated_per_step = []
    for al_step in range(1, args.al_steps + 1):
        # Acquisition: this should mutate the ALData object such that data_module.get_annot_data() now
        # contains the new labels as well.
        acquire_kwargs = {
            "al_step": al_step  # Used for keeping track of al_step in wandb for inner training loop of NP
        }

        pool_inds_annotated, extra_outputs = al_strategy.acquire_labels(classifier_module, **acquire_kwargs)
        if "time" in extra_outputs:
            logger.info("Acquisition time: {:.2f}s".format(extra_outputs["time"]))
        logger.info("Current data splits: {}".format(data_module))

        # Imbalance factors may be changed by data_module.imbalancer
        # NOTE: Could also be implemented using the same ALClassifier instance repeatedly, but in general
        #  different classifiers can be used at various points in the AL process.
        classifier_module = ALClassifier(
            args, data_module.classes, data_module.remaining_classes, data_module.imbalance_factors
        )
        annot_X, annot_y = data_module.get_annot_data().get_data()
        if args.wandb:
            wandb.log({f"labels_{al_step}": [num for _, num in extra_outputs["current_labels"].items()]})
            log_hist_of_labels(annot_y, al_step)

        # Train classifier
        extra_outputs = classifier_module.fit(annot_X, annot_y)
        # Evaluate classifier performance
        score, score_per_class = classifier_module.score(eval_data.X, eval_data.y, reduce="both")
        logger.info(
            "Step {}, score: {:.3f}, per class: [{}], gain: {:.3f}, gain per class: [{}]".format(
                al_step,
                score,
                ", ".join(f"{sc:.3f}" for sc in score_per_class),
                score - score_per_step[-1],
                ", ".join(f"{sc:.3f}" for sc in (score_per_class - score_per_class_per_step[-1])),
            )
        )
        if len(data_module.remaining_classes) == 2:  # Binary precision-recall
            class0_prec, class1_prec, class0_rec, class1_rec, pos_neg_dict = classifier_module.binary_prec_rec(
                eval_data.X, eval_data.y
            )
            logger.info(
                "Current precision-recall. Class 0: ({:.3f}, {:.3f}); Class 1: ({:.3f}, {:.3f})".format(
                    class0_prec, class0_rec, class1_prec, class1_rec
                )
            )
        logger.info("Classifier train time: {:.2f}s".format(extra_outputs["train_time"]))

        # Append all lists
        annotated_per_step.append(pool_inds_annotated.tolist())
        score_per_step.append(score)
        score_per_class_per_step.append(score_per_class)

        if args.wandb:
            wandb_dict = {eval_str + f"_class{cl}": sc for cl, sc in enumerate(score_per_class)}
            wandb_dict.update({"al_step": al_step, eval_str: score})
            if len(data_module.remaining_classes) == 2:  # Binary precision-recall
                wandb_dict.update(
                    {
                        "class0_prec": class0_prec,
                        "class0_rec": class0_rec,
                        "class1_prec": class1_prec,
                        "class1_rec": class1_rec,
                    }
                )
                wandb_dict.update(pos_neg_dict)

            if "oracle_improvements" in extra_outputs:
                wandb_dict.update({"oracle_improvements": wandb.Histogram(extra_outputs["oracle_improvements"])})
            if "f_scores" in extra_outputs:
                wandb_dict.update({"f_scores": wandb.Histogram(extra_outputs["f_scores"])})

            wandb.log(wandb_dict)

    # improvement_per_step = np.diff(score_per_step, n=1, axis=0)
    # improvement_per_class_per_step = np.diff(score_per_class_per_step, n=1, axis=0)

    logger.info(
        "FINISHED: Classifier score start: {:.3f}, per class: [{}]".format(
            score_per_step[0], ", ".join(f"{sc:.3f}" for sc in score_per_class_per_step[0])
        )
    )
    logger.info(
        "FINISHED: Classifier score end: {:.3f}, per class: [{}]".format(
            score_per_step[-1], ", ".join(f"{sc:.3f}" for sc in score_per_class_per_step[-1])
        )
    )
    if len(data_module.remaining_classes) == 2:  # Binary precision-recall
        logger.info(
            "FINISHED: Precision-Recall start: Class 0: ({:.3f}, {:.3f}); Class 1: ({:.3f}, {:.3f})".format(
                s_class0_prec, s_class0_rec, s_class1_prec, s_class1_rec
            )
        )
        logger.info(
            "FINISHED: Precision-Recall end: Class 0: ({:.3f}, {:.3f}); Class 1: ({:.3f}, {:.3f})".format(
                class0_prec, class0_rec, class1_prec, class1_rec
            )
        )


def main(args):
    # Folder structure is as follows:
    # out_dir
    #  data_type (e.g. mushrooms, mnist, etc.)
    #   acquisition strategy (e.g. oracle, random, etc.)
    #    date_string
    #     time_string
    #      - FILES_FROM_RUN
    #      DIRS_FROM_RUN
    date_string = f"{datetime.now():%Y-%m-%d}"
    time_string = f"{datetime.now():%H:%M:%S}"
    args.save_dir = args.out_dir / args.data_type / args.acquisition_strategy / date_string / time_string
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Console logger: print to stdout
    console = logging.StreamHandler()
    console.setLevel("INFO")
    console_formatter = logging.Formatter("%(name)-8s: %(levelname)-8s %(message)s", "%m-%d %H:%M:%S")
    console.setFormatter(console_formatter)
    # Debug logger: save to file
    debug = logging.FileHandler(args.save_dir / "debug.log")
    debug.setLevel("DEBUG")
    debug_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)-8s %(message)s", "%m-%d %H:%M:%S")
    debug.setFormatter(debug_formatter)
    # Add all this config to logger
    logging.basicConfig(level=logging.DEBUG, handlers=[console, debug])
    logger = logging.getLogger(__name__)

    # Disable any wandb requests from being printed to console
    logging.getLogger("requests").setLevel(logging.WARNING)  # Only log a message if it's at least a WARNING
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info("Arguments: \n{}".format(pformat(vars(args))))
    logger.info("Results will be saved to: {}".format(args.save_dir))

    # NOTE: Do all changes to args before here, so that wandb correctly logs them.
    if args.wandb:
        wandb_setup(args)

    do_active_learning_run(args, logger)


# if __name__ == "__main___":
args = create_arg_parser().parse_args()
assert args.resume_from_checkpoint is None, "resume_from_checkpoint not implemented yet."

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# Using torch 1.5.0 or 1.5.1 (apparently)
torch.cuda.manual_seed(args.seed)

# For reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

main(args)
