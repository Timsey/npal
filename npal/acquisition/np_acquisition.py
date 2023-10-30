import wandb
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from npal.utils import str2bool, str2none
from npal.data.data_module import DataWrapper

from npal.acquisition.np_modules.np_module import CNPModule
from npal.acquisition.base_acquisition import BaseStrategy


HEATMAP_RESOLUTION = 100


class NPStrategy(BaseStrategy):
    """
    Strategy that trains an NP on an oracle of the test-time AL setting that the strategy will be
    applied to (this is cheating).

    Note that this is trained on a true oracle if reward_split=args.eval_split. If reward_split="reward", then the
    NP is trained on the test-time AL setting, but using a biased objective (improvement of classifier on reward set,
    rather than on eval set).
    """

    def __init__(self, args, data_module, classifier_module, mode):
        super().__init__(args)
        self.mode = mode

        if mode == "np_oracle":
            self.name = "NP-Oracle"
            self.method = "np_oracle"
        elif mode == "np_future":
            self.name = "NP-Future"
            self.method = "np_resample_annot"
        else:
            raise RuntimeError("Unknown acquisition mode: {}.".format(mode))

        self.wandb = args.wandb
        self.np_acq_func = args.np_acq_func
        self.np_feature_type = args.np_feature_type
        self.eval_split = args.eval_split
        self.reward_split = args.reward_split
        self.do_val = args.np_do_val
        self.use_leave_one_out_rewards = args.use_leave_one_out_rewards
        self.logger = logging.getLogger(self.name)

        self.data_module = data_module
        self.classifier_module = classifier_module

        # For determining input dimensionality
        self.classes = data_module.classes
        dummy_X, dummy_y = data_module.get_annot_data().get_data()
        if len(dummy_y.shape) == 1:  # y is of shape (num_datapoints, ), make it (num_datapoints, 1)
            dummy_y = dummy_y[:, None]
        x_dim, y_dim = dummy_X.reshape(len(dummy_y), -1).shape[-1], dummy_y.shape[-1]
        self.cnp_module = CNPModule(args, x_dim, y_dim, data_module, classifier_module)

    def propose_acquisition(self, data_module, classifier_module, **kwargs):
        start_time = time.perf_counter()
        self.cnp_module.reset()  # Reset module every acquisition step

        # Create AL datasets given current data and classifier module
        self.logger.info("Generating AL datasets...")
        data_start = time.perf_counter()
        np_train, np_val, extra_data = self.cnp_module.create_np_datasets(
            data_module,
            classifier_module,
            mode=self.method,
            reward_split=self.reward_split,
            use_leave_one_out_rewards=self.use_leave_one_out_rewards,
            do_val=self.do_val,
            **kwargs,
        )
        data_time = time.perf_counter() - data_start
        self.logger.info("DONE: {:.2f}s".format(data_time))

        # Train the CNP
        self.logger.info("Training AL model...")
        train_start = time.perf_counter()
        self.cnp_module.train_cnp(np_train, np_val, **kwargs)
        train_time = time.perf_counter() - train_start
        self.logger.info("DONE: {:.2f}s".format(train_time))

        annot_data = data_module.get_annot_data()
        pool_data = data_module.get_pool_data()
        # Flattened version
        annot_data = DataWrapper(*annot_data.get_data(), flatten=True)
        pool_data = DataWrapper(*pool_data.get_data(), flatten=True)
        # Arrays of shape (num_pool_points, )
        means, stddevs = self.cnp_module.predict(annot_data, pool_data, **kwargs)

        extra_outputs = {
            "time": time.perf_counter() - start_time,
            "data_time": data_time,
            "train_time": train_time,
            "means": means,
            "stddevs": stddevs,
        }
        wandb_start = time.perf_counter()
        if self.wandb:
            al_step = kwargs["al_step"]
            oracle_improvements = None
            extra_plots = 0
            if "oracle_improvements" in extra_data:
                oracle_improvements = extra_data["oracle_improvements"]
                extra_plots = 1
            plt.figure(figsize=(6 + 6 * extra_plots, 5))
            # Subplot with stddev error bars
            plt.subplot(1, 2 + extra_plots, 2 + extra_plots)  # Create this one first, to get ylim for other subplots
            error_kw = dict(lw=1, capsize=0, capthick=0)
            plt.bar(list(range(len(means))), means, yerr=stddevs, label="pred", error_kw=error_kw)
            if "oracle_improvements" in extra_data:
                plt.bar(list(range(len(oracle_improvements))), oracle_improvements, label="oracle")
            plt.title("Improvement per action", fontsize=18)
            plt.xlabel("Action index", fontsize=15)
            plt.legend()
            _, _, ymin, ymax = plt.axis()
            # Subplot with only predictions
            plt.subplot(1, 2 + extra_plots, 1)  # Plot before previous, but use axes from previous
            plt.bar(list(range(len(means))), means, label="pred")
            plt.title("Improvement per action", fontsize=18)
            plt.ylabel("Improvement", fontsize=15)
            plt.xlabel("Action index", fontsize=15)
            plt.ylim(ymin, ymax)
            plt.legend()
            # Subplot without stddev error bars (but with oracle)
            if "oracle_improvements" in extra_data:  # Only if we have oracle improvements
                plt.subplot(1, 3, 2)  # Plot before previous, but use axes from previous
                plt.bar(list(range(len(means))), means, label="pred")
                plt.bar(list(range(len(oracle_improvements))), oracle_improvements, label="oracle")
                plt.title("Improvement per action", fontsize=18)
                plt.xlabel("Action index", fontsize=15)
                plt.ylim(ymin, ymax)
                plt.legend()
            # Log to wandb
            plt.tight_layout()
            wandb.log({f"np_improvements_step{al_step}": wandb.Image(plt)})
            plt.close()
            # Potentially also log heatmap
            wandb_time = time.perf_counter() - wandb_start
            extra_outputs["wandb_time"] = wandb_time
            wandb.log({f"np_wandb_log_time{al_step}": wandb_time})

        if self.np_acq_func == "mean":
            pool_inds_to_annotate = np.argsort(means)[-self.budget :]
        elif self.np_acq_func == "mean_plus_stddev":
            pool_inds_to_annotate = np.argsort(means + stddevs)[-self.budget :]
        else:
            raise ValueError("Unknown value for `np_acq_func`: {}".format(self.np_acq_func))

        return pool_inds_to_annotate, extra_outputs

    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # NP args
        parser.add_argument(
            "--np_r_dim", default=32, type=int, help="Dimensionality of feature encoding / representation."
        )
        parser.add_argument(
            "--np_attention",
            default="transformer",
            type=str2none,
            choices=[
                "multiplicative",
                "additive",
                "scaledot",
                "cosine",
                "manhattan",
                "euclidean",
                "weighted_dist",
                "multihead",
                "transformer",
                None,
            ],
            help="Attention type to use in CNP. If `None`, basic CNP is used.",
        )
        parser.add_argument(
            "--np_self_attention",
            default=None,
            type=str2none,
            choices=[
                "multiplicative",
                "additive",
                "scaledot",
                "cosine",
                "manhattan",
                "euclidean",
                "weighted_dist",
                "multihead",
                "transformer",
                None,
            ],
            help="Self attention to use. `None` if not used. Ignored if `attention` is `None`.",
        )

        # Training args
        parser.add_argument("--np_lr", default=0.001, type=float, help="Learning rate.")
        parser.add_argument(
            "--np_lr_gamma", default=0.1, type=float, help="Factor with which to decay lr during training."
        )
        parser.add_argument("--np_num_epochs", default=100, type=int, help="Number of epochs to train for.")
        parser.add_argument("--np_batch_size", default=64, type=int, help="Batch size to use.")

        # Data args
        parser.add_argument(
            "--reward_split",
            default="reward",
            type=str,
            choices=["reward", "val"],
            help=(
                "Which data split to use for computing rewards (note that using `val` or `test` is cheating, "
                "because it gives validation / test data info to the NP training)."
            ),
        )
        parser.add_argument(
            "--use_leave_one_out_rewards",
            default=False,
            type=str2bool,
            help="Whether to use leave-one-out rewards as y_cntxt (otherwise uses 0s).",
        )
        parser.add_argument(
            "--np_feature_type",
            default="raw",
            type=str,
            choices=["raw", "classifier"],
            help=(
                "Type of features to use for the NP. `raw` uses same features as used for classifier training."
                "`classifier` uses some classifier predictive features as well."
            ),
        )
        parser.add_argument(
            "--np_acq_func",
            default="mean",
            type=str,
            choices=["mean", "mean_plus_stddev"],
            help=(
                "Type of acquisition function to use on top of NP prediction. `mean` uses the max of the predicted "
                "mean. `mean_plus_stddev` uses the max of the sum of mean and stddev predictions. Note that we cannot "
                "use and UCB-like algorithm, since every `action` is only visited once at most during the AL process."
            ),
        )
        parser.add_argument(
            "--np_num_resampled_datasets",
            default=300,
            type=int,
            help="Number of resampled datasets to train on for non-oracle methods.",
        )
        parser.add_argument(
            "--project_np_features",
            default=False,
            type=str2bool,
            help=(
                "Whether to project NP input to `r_dim`. Usually used in conjunction with `np_feature_type` set to "
                "`classifier`, as this will combine data and classifier features in a higher dimensional space,"
                "then the original input space."
            ),
        )

        parser.add_argument(
            "--np_do_val",
            default=False,
            type=str2bool,
            help="Whether to construct NP validation data from the original annot-pool split.",
        )

        return parser
