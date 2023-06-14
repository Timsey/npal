import argparse
import pathlib

from npal.utils import str2bool, str2none, str2none_int
from npal.acquisition.heuristics import HALStrategy, CBalEntropyGreedyStrategy
from npal.acquisition.np_acquisition import NPStrategy


def create_arg_parser():
    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument(
        "--out_dir",
        default=None,
        type=pathlib.Path,
        help="Base path for saving results. Each run will create a subfolder.",
    )
    parser.add_argument(
        "--data_split_seed",
        default=42,
        type=int,
        help="Random seed to use for data splits. Ensures same train-val-test sets are uses throughout experiments.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed to use for everything but data splits.",
    )
    parser.add_argument(
        "--logging_level",
        default="INFO",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Level of logging.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        type=str2none,
        help="Checkpoint to resume from. Currently unused.",
    )

    # General data args
    parser.add_argument(
        "--abs_path_to_data_store",
        default=None,
        type=pathlib.Path,
        help="Path to dir where data is downloaded and/or stored.",
    )
    parser.add_argument(
        "--data_type",
        default="mushrooms",
        type=str,
        help="Data type to use.",
        choices=[
            "mushrooms",
            "waveform",
            "adult",
            "mnist",
            "fashion_mnist",
            "svhn",
            "cifar10",
        ],
    )
    parser.add_argument(
        "--eval_split",
        default="test",
        type=str,
        choices=["val", "test", "reward"],
        help="Which data split to use for evaluation of the AL strategy.",
    )
    parser.add_argument(
        "--total_num_points",
        default=None,
        type=str2none_int,
        help=(
            "Total number of data points to keep before making any splits (but after imbalancing). If a test data "
            "partition is provided then total_num_points * test_fraction test points are kept, and total_num_points * "
            "(1 - test_fraction) points are used to split into train, validation, etc. if possible. Else will raise "
            "an error."
        ),
    )
    parser.add_argument(
        "--test_fraction",
        default=0.2,
        type=float,
        help=(
            "Fraction of all data to use as test data. If a test data partition is provided then that data is "
            "used instead. In that case, this argument determines the relative number of test data points only if "
            "total_num_points is also set."
        ),
    )
    parser.add_argument(
        "--num_test_points",
        default=None,
        type=str2none_int,
        help=(
            "Number of points to use for testing. If `None`, test data split is determined by `test_fraction`, or "
            "by a separately provided test dataset. This argument is ignored if `total_num_points` is set.",
        ),
    )
    parser.add_argument(
        "--val_fraction",
        default=3 / 8,
        type=float,
        help="Fraction of all non-test data to use as validation data.",
    )
    parser.add_argument(
        "--reward_fraction",
        default=1 / 8,
        type=float,
        help="Fraction of all non-test data to use as external data for reward calculation.",
    )
    parser.add_argument(
        "--pool_fraction",
        default=0.9,
        type=float,
        help="Fraction of training data to consider as unlabeled on initialisation.",
    )
    parser.add_argument(
        "--num_train_points",
        default=None,
        type=str2none_int,
        help=(
            "Number of points to use for training. If `None`, train data is any leftover data after splitting of val, "
            "reward, and potentially test data. This argument is ignored if `total_num_points` is set.",
        ),
    )
    parser.add_argument(
        "--num_val_points",
        default=None,
        type=str2none_int,
        help=(
            "Number of points to use for validation. If `None`, validation data split is determined by `val_fraction`."
            "This argument is ignored if `total_num_points` is set.",
        ),
    )
    parser.add_argument(
        "--num_reward_points",
        default=None,
        type=str2none_int,
        help=(
            "Number of points to use for reward data. If `None`, reward data split is determined by "
            "`reward_fraction`. This argument is ignored if `total_num_points` is set.",
        ),
    )
    parser.add_argument(
        "--num_annot_points",
        default=None,
        type=str2none_int,
        help=(
            "Number of points to use for annot data. If `None`, reward data split is determined by "
            "1 - `pool_fraction`. This argument is ignored if `total_num_points` is set.",
        ),
    )
    parser.add_argument(
        "--num_pool_points",
        default=None,
        type=str2none_int,
        help=(
            "Number of points to use for pool data. If `None`, reward data split is determined by "
            "`pool_fraction`. This argument is ignored if `total_num_points` is set.",
        ),
    )
    parser.add_argument(
        "--stratify",
        default=True,
        type=str2bool,
        help="Whether to stratify data splits by class.",
    )
    parser.add_argument(
        "--imbalance_factors",
        default=[1, 1],
        nargs="+",
        type=float,
        help=(
            "Imbalancing factors to use: should be of length num_classes. Factors represent relative rarity "
            "(e.g. [4, 1] means class 0 appears 4x as often as class 1."
        ),
    )
    parser.add_argument(
        "--ignore_existing_imbalance",
        default=False,
        type=str2bool,
        help=(
            "Whether to ignore existing imbalance in the data. If `False` will aim to imbalance such that class "
            "rarities correspond to `imbalance_factors`. If `True`, will impose `imbalance_factors` on top of "
            "existing imbalance. Note that the latter may reduce the number of available data points."
        ),
    )
    parser.add_argument(
        "--normalise_features",
        default=True,
        type=str2bool,
        help="Whether to normalise data features.",
    )

    # AL args
    parser.add_argument(
        "--al_steps",
        default=10,
        type=int,
        help="Number of active learning steps to do.",
    )
    parser.add_argument(
        "--budget",
        default=1,
        type=int,
        help="Number of acquisitions to do per AL step.",
    )
    parser.add_argument(
        "--acquisition_strategy",
        default="random",
        type=str,
        choices=[
            "random",  # random sampling
            "oracle",  # myopic oracle
            "fscore",  # SVM minimum distance to hyperplane heuristic
            "uc_entropy",  # uncertainty sampling: entropy
            "uc_margin",  # uncertainty sampling: margin
            "uc_lc",  # uncertainty sampling: least confidence
            "hal_uni",  # Hybrid AL: trade-off between margin sampling and uniform diversity
            "hal_gauss",  # Hybrid AL: trade-off between margin sampling and gaussian diversity
            "kcgreedy",  # KC-greedy (i.e. sort of greedy CoreSet)
            "cbal",  # Class-Balanced Active Learning
            "np_oracle",  # myopic NP oracle (cheating; for debugging)
            "np_future",  # NP with resampling
            "gcn_random",  # random, but implemented by GCN code
            "gcn_unc",  # GCN method with uncertainty scores
            "gcn_core",  # GCN method with CoreSet scores
            "gcn_kcg",  # kcgreedy, but implemented by GCN paper code
            "gcn_lloss",  # Learning Loss (loss module), implemented by GCN paper code
            "gcn_vaal",  # Variational Adversarial Active Learning, implemented by GCN paper code
        ],
        help="Acquisition strategy to use for Active Learning.",
    )
    # Classifier args
    parser.add_argument(
        "--classifier_type",
        default="logistic",
        type=str,
        choices=["svm", "logistic", "cnn", "resnet"],
        help="Base classifier (task model) to use.",
    )
    parser.add_argument(
        "--class_weight_type",
        default="balanced",
        type=str2none,
        help=(
            "Which class weighting to use in the loss / score function. `None` does not perform weighting. "
            "`balanced` will set relative class weight equal to inverse of rarity weights (e.g. if a  "
            "class is 10 times rarer, it has 10 times the loss weight). `inverse` makes the rare class even "
            "more important by setting weights to the square of the rarity(e.g. if a "
            "class is 10 times rarer, it has 100 times the loss weight). Finally, set to e.g. `1-25` to use a custom "
            "weighting where class 1 is weighted 25x more than class 0 (dashes separate classes)."
        ),
    )
    parser.add_argument(
        "--use_class_weights_for_fit",
        default=False,
        type=str2bool,
        help=(
            "Whether to use class_weights during Sklearn classifier training as well. If `False` class_weights are "
            "only used for evaluation of the classifiers, mimicking the various utilities of the different classes."
        ),
    )
    parser.add_argument(
        "--score_type",
        default="accuracy",
        type=str,
        choices=["accuracy"],
        help="Score function to use for measuring classification performance.",
    )

    # wandb args
    parser.add_argument(
        "--wandb",
        default=False,
        type=str2bool,
        help="Whether to do wandb logging.",
    )
    parser.add_argument(
        "--wandb_project",
        default=None,
        type=str2none,
        help="Base project name to use for wandb logging.",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        type=str2none,
        help="Entity to use for wandb logging.",
    )

    # Add module specific args
    parser = NPStrategy.add_module_specific_args(parser)
    parser = HALStrategy.add_module_specific_args(parser)
    parser = CBalEntropyGreedyStrategy.add_module_specific_args(parser)

    return parser
