import copy
import wandb
import random
import logging
import collections
import time
import numpy as np
from functools import partial
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from npal.utils import count_parameters, count_trainable_parameters
from npal.data.data_module import DataWrapper

from .npf import CNPFLoss
from .npf.architectures import MLP, merge_flat_input
from npal.acquisition.acquisition_utils import compute_myopic_rewards

from .np_wrappers import MergeFeatureCNP, MergeFeatureAttnCNP


ANNOT_FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def map_nested_dicts_modify(ob, func):
    # Helper function for recursively applying a function to all
    # values in a nested dict (in-place)
    for k, v in ob.items():
        if isinstance(v, collections.abc.Mapping):
            map_nested_dicts_modify(v, func)
        else:
            ob[k] = func(v)


class GroupBatchSampler(Sampler):
    """
    Sampler that returns batches of indices, where every batch only contains indices of datasets with the same
    number of context and target points. This to ensure input-output to the NPs has the same shape within a batch.

    Alternative would be to zero-pad every dataset to the max size, but this requires adding 0s to y-values as well,
    and those are meaningful in general, so this is a bad idea.
    """

    def __init__(self, data_source, max_batch_size=16, shuffle=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        # Initialise batches
        self.batches = None
        self.reset_batches()

    def __iter__(self):
        if self.batches is None:
            raise RuntimeError("`batches` not yet initialised. Call `init_batches()` first.")
        for batch in self.batches:
            yield batch

    def __len__(self):
        if self.batches is None:
            raise RuntimeError("`batches` not yet initialised. Call `init_batches()` first.")
        return len(self.batches)

    def reset_batches(self):
        """Function to init batches, and also to shuffle batches between epochs."""
        batches = []
        # Group dataset indices that correspond to datasets of the same size. Create batches of maximum size given
        # by max_batch_size, and put the remainder in a separate batch. Shuffle dataset order within groups, so that
        # training happens on different(ly ordered) batches every epoch.
        for size_comb, indices in self.data_source.indices_per_size.items():
            if self.shuffle:
                random.shuffle(indices)  # Shuffle size_comb indices so that batches change between epochs
            num_batches = (
                len(indices) // self.max_batch_size
                if len(indices) % self.max_batch_size == 0
                else len(indices) // self.max_batch_size + 1
            )

            for k in range(num_batches):
                batches.append(indices[self.max_batch_size * k : self.max_batch_size * (k + 1)])
        if self.shuffle:
            random.shuffle(batches)  # Shuffle batches so that order is different every epoch
        self.batches = batches

    @staticmethod
    def group_collate_fn(data):
        # list of `whatever is returned by applying a batch in self.batches to the Dataset.__getitem__`, so in case
        # the batch in self.batches is a list of indices, this will be a list of length 1, containing only a tuple
        # that contains the actual data arrays, so what we actually want is data[0].
        X_cntxt, y_cntxt = [], []
        X_trgt, y_trgt = [], []
        for array_tuple in data[0]:
            # Each of these is of shape (num_annot/pool, num_features/1)
            annot_X, annot_y, pool_X, pool_y = array_tuple
            X_cntxt.append(annot_X)
            y_cntxt.append(annot_y)
            X_trgt.append(pool_X)
            y_trgt.append(pool_y)

        X_cntxt = torch.from_numpy(np.stack(X_cntxt)).float()
        y_cntxt = torch.from_numpy(np.stack(y_cntxt)).float()
        X_trgt = torch.from_numpy(np.stack(X_trgt)).float()
        y_trgt = torch.from_numpy(np.stack(y_trgt)).float()
        return X_cntxt, y_cntxt, X_trgt, y_trgt


class NPDataset(Dataset):
    def __init__(self, np_data):
        # np_data is a dict with annot and pool numbers as keys, where an item is a dict "annot" and "pool" as keys.
        # Each of those keys corresponds to a dict with X, y as keys. X, y themselves are arrays of
        # shape: (num_datasets, num_annot/num_pool, feature_dim/label_dim).
        # num_datasets should be the same for annot and pool data!
        # Here y_annot can either be 0 or equal to its own leave-one-out improvement.
        self.np_data = np_data
        # Flatten this dict such that every key corresponds to an index, storing indices of same-size datasets
        # In the sampler we will use indices_per_size to find the indices corresponding to a particular dataset size,
        # and then use dataset_per_index to find all the datasets of that size.
        self.indices_per_size = defaultdict(list)
        self.dataset_per_index = {}

        index = -1
        for annot_pool_size, annot_pool_dict in np_data.items():
            annot_X = annot_pool_dict["annot"]["X"]
            annot_y = annot_pool_dict["annot"]["y"]
            pool_X = annot_pool_dict["pool"]["X"]
            pool_y = annot_pool_dict["pool"]["y"]
            for i in range(len(annot_y)):  # All arrays have same first dim
                index += 1
                self.indices_per_size[annot_pool_size].append(index)
                self.dataset_per_index[index] = (annot_X[i], annot_y[i], pool_X[i], pool_y[i])

    def __len__(self):
        return len(self.dataset_per_index)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            assert isinstance(idx[0], int)
            return tuple([self.dataset_per_index[i] for i in idx])  # This will result is nested lists in the collate_fn
        elif isinstance(idx, int):
            return self.dataset_per_index[idx]
        else:
            raise RuntimeError("Incompatible index type {} in Dataset.__getitem__()".format(type(idx)))


class CNPModule:
    def __init__(self, args, x_dim, y_dim, data_module, classifier_module):
        self.args = args
        self.wandb = args.wandb
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.data_module = data_module
        self.classifier_module = classifier_module
        self.r_dim = args.np_r_dim
        self.attention = args.np_attention
        self.self_attention = args.np_self_attention
        self.lr = args.np_lr
        self.lr_gamma = args.np_lr_gamma
        self.batch_size = args.np_batch_size
        self.num_epochs = args.np_num_epochs

        self.np_feature_type = args.np_feature_type
        self.project_np_features = args.project_np_features
        self.logger = logging.getLogger("CNPModule")
        self.report_interval = self.num_epochs // 10  # Log every report_interval epochs to stdout

        self.num_resampled_datasets = args.np_num_resampled_datasets

        # If some imbalance factor is 0, then the underlying classifier will output only len(remaining_classes)
        #  number of predictions for every data point, which will not match len(classes).
        self.remaining_classes = data_module.remaining_classes
        if self.np_feature_type == "raw":
            num_extra_features = 0
        elif self.np_feature_type == "classifier":
            num_extra_features = classifier_module.classifier.get_num_classifier_features()
        else:
            raise ValueError("Unknown `np_feature_type` {}.".format(self.np_feature_type))
        self.num_extra_features = num_extra_features

        # Attributes set by self.reset()
        self.classifier_module = None
        self.reward_data = None
        self.use_leave_one_out_rewards = None
        self.norm_factor = None
        self.model = None
        self.loss_func = None
        self.optim = None
        self.scheduler = None
        self.device = None

    def reset(self):
        """
        Resets all object values and creates NP model using self.setup_cnp(). Call this every acquisition step!
        """

        # Below values are specific to (and will be filled in by) self.create_np_datasets()
        # Reward-related variables
        self.classifier_module = None
        self.reward_data = None
        self.use_leave_one_out_rewards = None

        # Global input data (test-time AL problem): necessary to store here for global normalisation
        self.norm_factor = None

        self.model, self.loss_func, self.optim, self.scheduler, self.device = self.setup_cnp()

    def setup_cnp(self):
        # MODEL SETUP
        XEncoder = partial(MLP, n_hidden_layers=1, hidden_size=self.r_dim)
        # MLP takes single input but we give x and R so merge them
        Decoder = merge_flat_input(partial(MLP, n_hidden_layers=4, hidden_size=self.r_dim), is_sum_merge=True)
        # MLP takes single input but we give x and y so merge them
        XYEncoder = merge_flat_input(
            partial(MLP, n_hidden_layers=2, hidden_size=self.r_dim * 2),
            is_sum_merge=True,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        np_kwargs = {
            "encoded_path": "deterministic",
            "x_transf_dim": self.r_dim,
            "num_extra_features": self.num_extra_features,
            "project_np_features": self.project_np_features,
            "is_heteroskedastic": True,
            "XEncoder": XEncoder,
            "Decoder": Decoder,
        }
        if self.attention is not None:
            np_kwargs.update(
                {
                    "attention": self.attention,
                    "attention_kwargs": {},
                    "is_self_attn": self.self_attention is not None,
                    "self_attention_kwargs": {
                        "attention": self.self_attention,
                        # TODO: These don't work because SelfAttention.forward() does not receive a `positions` arg
                        #  through the merge_flat_input() wrapper.
                        # "positional": "absolute",
                        # "position_dim": 1,
                    },
                }
            )
            cnp = MergeFeatureAttnCNP(
                x_dim=self.x_dim,
                y_dim=self.y_dim,
                r_dim=self.r_dim,
                XYEncoder=XYEncoder,  # Not used if `is_self_attn`
                **np_kwargs,
            ).to(device)
            num_params = count_parameters(cnp)
            num_trainable_params = count_trainable_parameters(cnp)
            self.logger.info(
                "ATTNCNP with {} parameters, of which {} trainable.".format(num_params, num_trainable_params)
            )
        else:
            cnp = MergeFeatureCNP(
                x_dim=self.x_dim,
                y_dim=self.y_dim,
                r_dim=self.r_dim,
                XYEncoder=XYEncoder,
                **np_kwargs,
            ).to(device)
            num_params = count_parameters(cnp)
            num_trainable_params = count_trainable_parameters(cnp)
            self.logger.info("CNP with {} parameters, of which {} trainable.".format(num_params, num_trainable_params))

        # Optimiser and scheduler
        optim = torch.optim.Adam(cnp.parameters(), lr=self.lr, weight_decay=0)
        lr_gamma = self.lr_gamma ** (1 / self.num_epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, lr_gamma)

        # is_force_mle_eval doesn't do anything unless using latent path
        # reduction sums over probs for a single dataset (i.e. function), so the loss will be more extreme if there
        #  are many target points. This is why we do no reduction.
        loss_func = CNPFLoss(reduction=None, is_force_mle_eval=False)
        return cnp, loss_func, optim, scheduler, device

    def create_np_datasets(
        self,
        data_module,
        classifier_module,
        mode="resample_annot",
        reward_split="reward",
        use_leave_one_out_rewards=False,
        do_val=False,  # Only used for `resample_annot` for now.
        **kwargs,
    ):
        """
        Creates AL datasets to train NP on. Populates properties such as self.classifier_module, self.reward_data,
        and self.use_leave_one_out_rewards, which are specific to a dataset setting.

        mode: data generating mode.
         - "resample_annot": create datasets by re-sampling the annotated part of meta-train. "Future generalisation"
         - "oracle": use the current annot-pool setting as the (only!) training setting. Mostly just a sanity check.

        reward_split: reward generating data.
         - "reward": use the reward set for computing rewards.
         - "val": use the validation set for computing rewards. [cheating if eval_set = val]
         - "test": use the test set for computing rewards. [cheating if eval_set = test]
        """

        self.classifier_module = classifier_module
        self.use_leave_one_out_rewards = use_leave_one_out_rewards

        extra_data = {}

        # Data split to use for calculating reward
        if reward_split == "val":
            self.reward_data = data_module.get_val_data()
        elif reward_split == "test":
            self.reward_data = data_module.get_test_data()
        elif reward_split == "reward":
            self.reward_data = data_module.get_reward_data()
        else:
            raise ValueError(f"Unknown value for `reward_split`: {reward_split}.")

        # Actual test-time AL setting
        annot_data = data_module.get_annot_data()
        pool_data = data_module.get_pool_data()
        # Normalise between [-1, 1]: store normalisation factor here for global normalisation.
        # Used in self.compute_np_features()
        self.norm_factor = max(annot_data.get_max_of_abs(), pool_data.get_max_of_abs())

        if mode == "np_resample_annot":
            np_val = None
            if do_val:
                # Get the test-time dataset as np-val. This is the data that we can use to keep track of
                #  performance on the test-time annot-pool problem, but note that this uses pool data labels.
                np_val, extra_data = self.create_test_time_dataset(annot_data, pool_data, extra_data)

            # np_train is created from resampling the annot data
            resampled_annot_inds = []
            resampled_pool_inds = []
            resampled_sizes = []
            num_annot = len(annot_data)

            annot_X, annot_y = annot_data.get_data()
            for _ in range(self.num_resampled_datasets):
                # Get sizes for this resampled annot-pool dataset
                # num_annot_resampled = np.random.choice(np.arange(1, num_annot))
                # TODO: Make this less random / more equal distribution over sizes / bias towards more useful sizes?
                num_annot_resampled = int(np.random.choice(ANNOT_FRACS) * num_annot)
                num_pool_resampled = num_annot - num_annot_resampled
                # Get indices for annot and pool resamplings
                # Note: Have to select indices such that at least two classes are present in annotated
                inds_of_resampled_annot = None
                i = 0
                # Need to make sure that our sampled AL datasets contain more than 1 class, so do a rejection step here.
                while inds_of_resampled_annot is None:
                    if i > 1000:
                        # NOTE: with imbalance factor 9, there is a 0.1 prob of choosing rare class on any selection.
                        #  If we do num_annot_resampled * 100 selections in total. Prob of not choosing any rare class
                        #  is < 2.7 * 10e-5 when num_annot_resampled = 1 (and already << 10e-6 for = 2). Since we in
                        #  fact do 10x that many selections, this should never happen.
                        raise RuntimeError("Too many rejections for resampling annotated data.")
                    i += 1
                    suggested_inds = np.random.choice(range(num_annot), size=num_annot_resampled, replace=False)
                    classes = set(annot_y[suggested_inds])
                    if len(classes) > 1:  # Else: try again
                        if self.use_leave_one_out_rewards:
                            # In this case, we need either 3+ classes, or 2 examples of each class, otherwise we
                            # cannot calculate leave-one-out rewards for all sampled settings.
                            if len(classes) > 2:  # 3+ classes?
                                inds_of_resampled_annot = suggested_inds
                            else:  # 2 classes, so check that there are at least 2 examples of each class
                                class0_num = np.sum((annot_y[suggested_inds] == list(classes)[0]))
                                class1_num = np.sum((annot_y[suggested_inds] == list(classes)[1]))
                                if class0_num > 1 and class1_num > 1:
                                    inds_of_resampled_annot = suggested_inds
                        else:  # Having at least 2 classes is sufficient
                            inds_of_resampled_annot = suggested_inds
                # Pool is all other inds
                inds_of_resampled_pool = np.array(
                    [ind for ind in range(num_annot) if ind not in inds_of_resampled_annot]
                )

                resampled_sizes.append((num_annot_resampled, num_pool_resampled))
                resampled_annot_inds.append(inds_of_resampled_annot)
                resampled_pool_inds.append(inds_of_resampled_pool)
            assert (
                len(resampled_annot_inds) == self.num_resampled_datasets
            ), "Missing some resampled datasets? Found {}, but expected {}.".format(
                len(resampled_annot_inds), self.num_resampled_datasets
            )

            np_train = {}
            for i in range(self.num_resampled_datasets):
                # Bit hacky, but np_input functions require DataWrapper.get_data(). Also can flatten here.
                a_data = DataWrapper(
                    annot_X[resampled_annot_inds[i], ...], annot_y[resampled_annot_inds[i]], flatten=True
                )
                p_data = DataWrapper(
                    annot_X[resampled_pool_inds[i], ...], annot_y[resampled_pool_inds[i]], flatten=True
                )
                # shape = (num_datapoints, features/1)
                annot_features, annot_improvements, pool_features, pool_improvements = self.compute_np_input_for_train(
                    a_data, p_data
                )
                resampled_size = resampled_sizes[i]
                if resampled_size not in np_train:
                    # Add current resampled annot-pool size to data dict
                    # [None, ...] to add dataset dimension: (num_datasets, num_annot/pool, num_features/1)
                    np_train[resampled_size] = {
                        "annot": {
                            "X": [annot_features],
                            "y": [annot_improvements],  # leave-one-out improvements OR 0
                        },
                        "pool": {
                            "X": [pool_features],
                            "y": [pool_improvements],  # improvements
                        },
                    }
                else:
                    # Append current resampled annot-pool setting to pre-existing entry
                    np_train[resampled_size]["annot"]["X"].append(annot_features)
                    np_train[resampled_size]["annot"]["y"].append(annot_improvements)
                    np_train[resampled_size]["pool"]["X"].append(pool_features)
                    np_train[resampled_size]["pool"]["y"].append(pool_improvements)
            # Stack arrays in nested dict
            map_nested_dicts_modify(np_train, np.stack)

        elif mode == "np_oracle":
            # For oracle method, both np-train and np-val are the test-time annot-pool setting.
            # NOTE: This may use meta-reward instead of meta-validation improvements (i.e. when
            #  `reward_split` != "val"). In that case, this is a slightly biased oracle (it picks the point that is
            #  best on meta-reward data, rather than meta-validation data). This makes it closer to the non-oracle
            #  methods in principle.
            np_train, extra_data = self.create_test_time_dataset(annot_data, pool_data, extra_data)
            # Train and validation are the same (both the actual setting)
            np_val = copy.deepcopy(np_train)
        else:
            raise ValueError("Unknown data generating `mode`: {}.".format(mode))

        return np_train, np_val, extra_data

    def train_cnp(self, np_train, np_val, **kwargs):
        """Train the CNP on the annot-pool re-samplings, and evaluate on the full annot-pool."""

        al_step = kwargs["al_step"]
        # NOTE: No test data, since we are just training the NP here: we only care about test performance
        #  of the complete AL system
        train_loader = self.create_np_dataloader(np_train, self.batch_size, shuffle=True)

        # Initial evaluation
        wandb_dict = {}
        if np_val:
            val_start = time.perf_counter()
            val_loader = self.create_np_dataloader(np_val, self.batch_size * 2, shuffle=False)
            val_loss = self.val_epoch(self.model, val_loader, self.loss_func, self.device)
            val_time = time.perf_counter() - val_start
            log_str = "{:>10} {:>24} {:>23} {:>22} {:>21}".format(
                f"Epoch {0:>3}",
                f"train loss: {'N/A':>8}",
                f"train time: {'N/A':>7}",
                f"val loss: {val_loss:>8.3f}",
                f"val time: {val_time:>6.2f}s",
            )
            if self.wandb:
                wandb_dict.update(
                    {
                        f"np_val_loss_als{al_step}": val_loss,
                        f"np_epoch_als{al_step}": 0,
                    }
                )
                wandb.log(wandb_dict)
        else:
            log_str = "{:>10} {:>24} {:>23}".format(
                f"Epoch {0:>3}",
                f"train loss: {'N/A':>8}",
                f"train time: {'N/A':>7}",
            )
        self.logger.info(log_str)

        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            wandb_dict = {}
            if self.wandb:
                wandb_dict.update({f"np_lr_als{al_step}": self.optim.param_groups[0]["lr"]})

            train_start = time.perf_counter()
            train_loss = self.train_epoch(
                self.model,
                train_loader,
                self.loss_func,
                self.optim,
                self.scheduler,
                self.device,
            )
            train_time = time.perf_counter() - train_start

            if np_val:
                val_start = time.perf_counter()
                val_loss = self.val_epoch(
                    self.model,
                    val_loader,
                    self.loss_func,
                    self.device,
                )
                val_time = time.perf_counter() - val_start

                log_str = "{:>10} {:>24} {:>23} {:>22} {:>21}".format(
                    f"Epoch {epoch:>3}",
                    f"train loss: {train_loss:>8.3f}",
                    f"train time: {train_time:>6.2f}s",
                    f"val loss: {val_loss:>8.3f}",
                    f"val time: {val_time:>6.2f}s",
                )
                if self.wandb:
                    wandb_dict.update(
                        {
                            f"np_train_loss_als{al_step}": train_loss,
                            f"np_val_loss_als{al_step}": val_loss,
                            f"np_epoch_als{al_step}": epoch,
                        }
                    )
                    wandb.log(wandb_dict)
            else:
                log_str = "{:>10} {:>24} {:>23}".format(
                    f"Epoch {epoch:>3}",
                    f"train loss: {train_loss:>8.3f}",
                    f"train time: {train_time:>6.2f}s",
                )
                if self.wandb:
                    wandb_dict.update(
                        {
                            f"np_train_loss_als{al_step}": train_loss,
                            f"np_epoch_als{al_step}": epoch,
                        }
                    )
                    wandb.log(wandb_dict)
            if epoch % self.report_interval == 0:
                self.logger.info(log_str)
            else:
                self.logger.debug(log_str)

    def predict(self, annot_data, pool_data, **kwargs):
        """
        Do a forward pass of the CNP for a single dataset of annot-pool points.
        """

        annot_features, annot_improvements, pool_features = self.compute_np_input_for_predict(annot_data, pool_data)

        with torch.no_grad():
            self.model.eval()
            # Unsqueeze adds dataset dimension; we only have a single annot-pool setting here.
            X_cntxt = torch.from_numpy(annot_features).unsqueeze(0).float().to(self.device)
            y_cntxt = torch.from_numpy(annot_improvements).unsqueeze(0).float().to(self.device)
            X_trgt = torch.from_numpy(pool_features).unsqueeze(0).float().to(self.device)
            # p_yCc (p_y_trgt), z_samples, q_zCc, q_zCct
            pred_dist = self.model.forward(X_cntxt, y_cntxt, X_trgt, Y_trgt=None)[0]  # y_trgt only used for latent path
            # pred_dist contains tensors of shape (1, num_datasets=1, num_pool_points, 1): remove the singular dims.
            mean = pred_dist.mean[0, 0, :, 0].cpu().numpy()
            stddev = pred_dist.stddev[0, 0, :, 0].cpu().numpy()
        return mean, stddev

    def create_test_time_dataset(self, annot_data, pool_data, extra_data):
        """
        Compute input on the full test-time problem. Essentially the starting annot-pool setting.
        This is both np-train and np-val for this oracle method, but only np-val for the other methods.
        Note that using this dataset for np-val is more for logging purposes, since the actual test-time problem uses
        the meta-test set for computing scores on np-val acquisitions, rather than the (meta-)reward set.

        Note that in doing this we are not overfitting to the AL problem (meta-val or meta-test), since this np-val is
        constructed only from meta-train data (e.g. annot-pool). However, in real-life cases we do not actually have
        access to pool labels of meta-train, so np-val does include info that we do not have at test time, and np-val
        should not be used to tune the method per-dataset / classifier (only for determining initial general settings).
        """

        # Get same data, but flattened
        annot_data = DataWrapper(*annot_data.get_data(), flatten=True)
        pool_data = DataWrapper(*pool_data.get_data(), flatten=True)
        annot_features, annot_improvements, pool_features, pool_improvements = self.compute_np_input_for_train(
            annot_data, pool_data
        )
        # Pool is test-time setting in this case
        extra_data["oracle_improvements"] = pool_improvements[:, 0]  # Remove extra dim

        # [None, ...] to add dataset dimension: (num_datasets, num_annot/pool, num_features/1)
        test_time_dataset = {
            (len(annot_data), len(pool_data)): {
                "annot": {
                    "X": annot_features[None, ...],
                    "y": annot_improvements[None, ...],  # leave-one-out improvements
                },
                "pool": {
                    "X": pool_features[None, ...],
                    "y": pool_improvements[None, ...],  # improvements
                },
            }
        }
        return test_time_dataset, extra_data

    def compute_np_input_for_predict(self, annot_data, pool_data):
        """
        Function to compute context and target values for prediction (i.e. no target_y), for a given annot and
        pool set + classifier and reward data setting.
        """

        annot_features, pool_features = self.compute_np_features(annot_data, pool_data)
        if self.use_leave_one_out_rewards:
            annot_improvements = self.compute_leave_one_out_rewards(annot_data)
        else:
            # Just use 0s for y_cntxt (= annot_improvements)
            annot_improvements = np.zeros((len(annot_data), 1))
        return annot_features, annot_improvements, pool_features

    def compute_np_input_for_train(self, annot_data, pool_data):
        """
        Function to compute context and target values for training (i.e. including target_y), for a given annot and
        pool set + classifier and reward data setting.
        """
        annot_features, annot_improvements, pool_features = self.compute_np_input_for_predict(annot_data, pool_data)
        pool_improvements = self.compute_pool_rewards(annot_data, pool_data)
        return annot_features, annot_improvements, pool_features, pool_improvements

    def compute_leave_one_out_rewards(self, leave_one_out_data):
        start_time = time.perf_counter()
        leave_one_out_X, leave_one_out_y = leave_one_out_data.get_data()
        # Create index arrays for annotated (all-but-one index) and pool (one index)
        num_points = len(leave_one_out_y)
        annot_idx_groups = np.arange(1, num_points) - np.tri(num_points, num_points - 1, k=-1, dtype=bool)
        pool_idx_groups = np.arange(0, num_points)
        # Loop over leave-one-out sets
        leave_one_out_improvements = []
        for annot_inds, pool_idx in zip(annot_idx_groups, pool_idx_groups):
            annot_X = leave_one_out_X[annot_inds, ...]
            annot_y = leave_one_out_y[annot_inds]
            # Slice to retain shape
            pool_X = leave_one_out_X[pool_idx : pool_idx + 1, ...]
            pool_y = leave_one_out_y[pool_idx : pool_idx + 1]
            improvement, _ = compute_myopic_rewards(
                self.classifier_module,
                self.reward_data,
                annot_X,
                annot_y,
                pool_X,
                pool_y,
                report_interval=None,
                logger=self.logger,
            )
            leave_one_out_improvements.append(improvement[0])  # improvement is list on len 1
        self.logger.debug("Leave-one-out rewards time: {:.2f}s".format(time.perf_counter() - start_time))
        # shape = (num_annot/pool, 1)
        return np.array(leave_one_out_improvements)[:, None]

    def compute_pool_rewards(self, annot_data, pool_data):
        annot_X, annot_y = annot_data.get_data()  # Automatically gets feature data instead of raw
        pool_X, pool_y = pool_data.get_data()  # Automatically gets feature data instead of raw
        pool_improvements, time_taken = compute_myopic_rewards(
            self.classifier_module,
            self.reward_data,
            annot_X,
            annot_y,
            pool_X,
            pool_y,
            report_interval=None,
            logger=self.logger,
        )
        self.logger.debug("Pool rewards time: {:.2f}s".format(time_taken))
        # shape = (num_annot/pool, 1)
        # NOTE: We've check that this gives the same scores as Oracle does when applied to test-time setting (with
        #  eval_split == reward_split).
        return np.array(pool_improvements)[:, None]

    def compute_np_features(self, annot_data, pool_data):
        annot_F, annot_y = annot_data.get_data()
        pool_F, pool_y = pool_data.get_data()
        # NOTE: Not using labels (_y) here for context data at all. Future work.
        # NOTE: All kinds of features are possible: even those correlating annot and pool data?
        # Global normalisation
        # NOTE: This normalisation is slightly different than the one used for classifier features.
        annot_F = annot_F / self.norm_factor
        pool_F = pool_F / self.norm_factor
        if self.np_feature_type == "raw":
            pass  # Do nothing beyond normalisation
        elif self.np_feature_type == "classifier":
            # E.g. probs for LogisticRegression or f-scores for SVM
            # shape = (num_points x num_classes)
            annot_C = self.classifier_module.get_classifier_features_for_np(annot_F)
            pool_C = self.classifier_module.get_classifier_features_for_np(pool_F)
            # Normalise to [-1, 1] for NP
            feature_max = max(np.max(annot_C), np.max(pool_C))
            feature_min = min(np.min(annot_C), np.min(pool_C))
            # NOTE: If this gives divide-by-zero then something is wrong with the classifier probably.
            annot_C = (annot_C - feature_min) / (feature_max - feature_min) * 2 - 1
            pool_C = (pool_C - feature_min) / (feature_max - feature_min) * 2 - 1
            # Concatenate together: shape (num_points x num_data_features + num_classes (e.g. f-score, or prob, etc).
            annot_F = np.concatenate((annot_F, annot_C), axis=1)
            pool_F = np.concatenate((pool_F, pool_C), axis=1)
        else:
            raise ValueError("Unknown `np_features` value: {}".format(self.np_feature_type))

        # shape = (num_annot/pool, num_features)
        return annot_F, pool_F

    @staticmethod
    def create_np_dataloader(np_data, batch_size, shuffle):
        # NPDataset structures the np_data dict, such that GroupBatchSampler returns batches of indices all
        # corresponding to datasets of the same size (within in a batch).
        dataset = NPDataset(np_data)
        sampler = GroupBatchSampler(dataset, max_batch_size=batch_size, shuffle=shuffle)
        loader = DataLoader(dataset, sampler=sampler, collate_fn=sampler.group_collate_fn)
        return loader

    @staticmethod
    def train_epoch(model, loader, loss_func, optim, scheduler, device):
        """Epoch on re-sampled annot-pool (in meta-train)."""
        model.train()
        loss_func.train()
        epoch_loss = 0
        datasets_in_epoch = 0
        for batch in loader:
            X_cntxt, y_cntxt, X_trgt, y_trgt = batch
            X_cntxt = X_cntxt.to(device)
            y_cntxt = y_cntxt.to(device)
            X_trgt = X_trgt.to(device)
            y_trgt = y_trgt.to(device)
            optim.zero_grad()
            # pred_outputs = p_yCc (p_y_trgt), z_samples, q_zCc, q_zCct
            pred_outputs = model.forward(X_cntxt, y_cntxt, X_trgt, Y_trgt=None)  # y_trgt only used for latent path
            non_red_loss = loss_func(pred_outputs, y_trgt)
            # average over target points (non_red_loss is already summed over target points, so divide by number of
            # target points, and then sum over datasets.
            sum_loss = (non_red_loss / y_trgt.shape[1]).sum(dim=0)
            loss = sum_loss / len(X_cntxt)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optim.step()
            epoch_loss += sum_loss
            datasets_in_epoch += len(X_cntxt)

        if scheduler is not None:
            scheduler.step()
        epoch_loss /= datasets_in_epoch
        loader.sampler.reset_batches()  # Reset batches for training, so that next epoch has different data ordering
        return epoch_loss.item()

    @staticmethod
    def val_epoch(model, loader, loss_func, device):
        """Epoch on full annot-pool (in meta-train)."""
        with torch.no_grad():
            model.eval()
            loss_func.eval()
            epoch_loss = 0
            datasets_in_epoch = 0
            for i, batch in enumerate(loader):
                X_cntxt, y_cntxt, X_trgt, y_trgt = batch
                X_cntxt = X_cntxt.to(device)
                y_cntxt = y_cntxt.to(device)
                X_trgt = X_trgt.to(device)
                y_trgt = y_trgt.to(device)
                # pred_outputs = p_yCc (p_y_trgt), z_samples, q_zCc, q_zCct
                pred_outputs = model.forward(X_cntxt, y_cntxt, X_trgt, Y_trgt=None)  # y_trgt only used for latent path
                non_red_loss = loss_func(pred_outputs, y_trgt)
                # average over target points (non_red_loss is already summed over target points, so divide by number of
                # target points, and then sum over datasets.
                sum_loss = (non_red_loss / y_trgt.shape[1]).sum(dim=0)
                epoch_loss += sum_loss
                datasets_in_epoch += len(X_cntxt)

            epoch_loss /= datasets_in_epoch
            return epoch_loss.item()
