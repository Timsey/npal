"""
Copied and adapted from: https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning
"""

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from npal.acquisition.base_acquisition import BaseStrategy
from npal.acquisition.gcn_architecture.gcn_models import VAE, Discriminator, GCN
from npal.acquisition.gcn_architecture.gcn_helpers import (
    aff_to_adj,
    BCEAdjLoss,
    train_vaal,
    get_features,
    get_kcg,
    get_uncertainty,
    kCenterGreedy,
)


CUDA_VISIBLE_DEVICES = 0
# NUM_TRAIN = 50000  # N \Fashion MNIST 60000, cifar 10/100 50000
# NUM_VAL = 50000 - NUM_TRAIN
BATCH = 128  # B
MAX_POOL_SIZE = 10000  # M  (used to be called `SUBSET`)
# ADDENDUM = 1000  # K; budget size, I think?

# TRIALS = 5
# CYCLES = 5

# EPOCH_GCN = 200
LR_GCN = 1e-3
WDECAY = 5e-4  # 2e-3# 5e-4

# Argparse defaults: 02-MAY-2022
# https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning/blob/d8993bcaab8e5de7c6a581d2a9c803fdad0470ce/main.py
HIDDEN_UNITS = 128
DROPOUT_RATE = 0.3
LAMBDA_LOSS = 1.2
S_MARGIN = 0.1


class GCNStrategy(BaseStrategy):
    """
    Random strategy: label points uniformly at random. This is the default ML setting.
    """

    def __init__(self, args, data_module, classifier_module, mode):
        super().__init__(args)
        self.mode = mode
        self.data_type = args.data_type

        if mode == "gcn_random":
            self.name = "GCN-Random"
            self.method = "Random"
        elif mode == "gcn_unc":
            self.name = "GCN-Uncertain"
            self.method = "UncertainGCN"
        elif mode == "gcn_core":
            self.name = "GCN-Core"
            self.method = "CoreGCN"
        elif mode == "gcn_kcg":
            self.name = "GCN-KC-Greedy"
            self.method = "CoreSet"
        elif mode == "gcn_lloss":
            self.name = "GCN-Learning-Loss"
            self.method = "lloss"
        elif mode == "gcn_vaal":
            self.name = "GCN-VAAL"
            self.method = "VAAL"
        else:
            raise RuntimeError("Unknown acquisition mode: {}.".format(mode))

    def reset(self, data_module, classifier_module):
        # Wrap current classifier
        nn_classifier = classifier_module.classifier.model
        if not isinstance(nn_classifier, nn.Module):
            raise RuntimeError(
                "GCNStrategy only supports torch.nn.Module classifier models, not {}".format(type(nn_classifier))
            )
        models = {"backbone": nn_classifier}  # Forward of this only does base model forward
        if self.method == "lloss":
            models.update({"module": nn_classifier.lloss})  # Separate forward method for this

        # Wrap current data
        annot_data = data_module.get_annot_data()
        pool_data = data_module.get_pool_data()
        dataset = GCNDataWrapper(annot_data, pool_data)  # Object containing all data
        # Construct indices for labeled and unlabeled data. Unlabeled has lowest indices.
        unlabeled_set = list(range(len(pool_data)))
        labeled_set = list(range(len(pool_data), len(pool_data) + len(annot_data)))
        # Create subset of unlabeled data; shuffle indices of unlabeled first
        random.shuffle(unlabeled_set)
        subset = unlabeled_set[:MAX_POOL_SIZE]  # Subset of unlabeled data inds, corresponding to pool data in `dataset`
        return models, dataset, labeled_set, subset

    def propose_acquisition(self, data_module, classifier_module, **kwargs):
        start_time = time.perf_counter()
        # cycle = kwargs["al_step"] - 1  # we start at 1, they start at 0

        # Setup classifier and data
        models, dataset, labeled_set, subset = self.reset(data_module, classifier_module)

        # Do query
        # Returns low-to-high indices of `subset` on whatever metric used; i.e. high is better.
        sorted_subset_indices = query_samples(
            self.data_type, models, self.method, dataset, subset, labeled_set, self.budget
        )
        # Get the `budget` best indices, and apply to `subset` to find actual pool indices.
        # Since pool indices come first in all indices of `dataset`, these correspond to the actual pool indices.
        pool_inds_to_annotate = np.array(subset)[sorted_subset_indices[-self.budget :]]

        extra_outputs = {
            "time": time.perf_counter() - start_time,
        }
        return pool_inds_to_annotate, extra_outputs

    # @staticmethod
    # def add_module_specific_args(parent_parser):
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #
    #     # NP args
    #     parser.add_argument("--lloss", default=False, type=str2bool, help="Whether to do learning loss.")
    #
    #     return parser


# Below is copied and adapted from: https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning
def query_samples(data_type, model, method, all_data, subset, labeled_set, addendum):
    if method == "Random":
        arg = np.random.randint(len(subset), size=len(subset))

    if (method == "UncertainGCN") or (method == "CoreGCN"):
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(
            all_data,
            batch_size=BATCH,
            sampler=SubsetSequentialSampler(subset + labeled_set),
            # more convenient if we maintain the order of subset
            pin_memory=True,
        )
        binary_labels = torch.cat((torch.zeros([len(subset), 1]), (torch.ones([len(labeled_set), 1]))), 0)

        features = get_features(model, unlabeled_loader)
        features = nn.functional.normalize(features)
        adj = aff_to_adj(features)

        gcn_module = GCN(nfeat=features.shape[1], nhid=HIDDEN_UNITS, nclass=1, dropout=DROPOUT_RATE).cuda()

        models = {"gcn_module": gcn_module}

        optim_backbone = optim.Adam(models["gcn_module"].parameters(), lr=LR_GCN, weight_decay=WDECAY)
        optimizers = {"gcn_module": optim_backbone}

        # Indices of labeled data in `features` (or rows/columns of `adj`)
        lbl = np.arange(len(subset), len(subset) + len(all_data.annot_data), 1)
        # Indices of unlabeled data in `features` (or rows/columns of `adj`)
        nlbl = np.arange(0, len(subset), 1)

        ############
        for _ in range(200):
            optimizers["gcn_module"].zero_grad()
            outputs, _, _ = models["gcn_module"](features, adj)
            lamda = LAMBDA_LOSS
            loss = BCEAdjLoss(outputs, lbl, nlbl, lamda)
            loss.backward()
            optimizers["gcn_module"].step()

        models["gcn_module"].eval()
        with torch.no_grad():
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = features.cuda()
                labels = binary_labels.cuda()
            scores, _, feat = models["gcn_module"](inputs, adj)

            if method == "CoreGCN":
                feat = feat.detach().cpu().numpy()
                new_av_idx = np.arange(len(subset), len(subset) + len(all_data.annot_data))
                sampling2 = kCenterGreedy(feat)
                batch2 = sampling2.select_batch_(new_av_idx, addendum)
                other_idx = [x for x in range(len(subset)) if x not in batch2]
                arg = other_idx + batch2
            else:
                s_margin = S_MARGIN
                scores_median = np.squeeze(torch.abs(scores[: len(subset)] - s_margin).detach().cpu().numpy())
                arg = np.argsort(-(scores_median))

            print("Max confidence value: ", torch.max(scores.data))
            print("Mean confidence value: ", torch.mean(scores.data))
            preds = torch.round(scores)
            correct_labeled = (preds[len(subset) :, 0] == labels[len(subset) :, 0]).sum().item() / len(
                all_data.annot_data
            )
            correct_unlabeled = (preds[: len(subset), 0] == labels[: len(subset), 0]).sum().item() / len(subset)
            correct = (preds[:, 0] == labels[:, 0]).sum().item() / (len(subset) + len(all_data.annot_data))
            print("Labeled classified: ", correct_labeled)
            print("Unlabeled classified: ", correct_unlabeled)
            print("Total classified: ", correct)

    if method == "CoreSet":
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(
            all_data,
            batch_size=BATCH,
            sampler=SubsetSequentialSampler(subset + labeled_set),
            # more convenient if we maintain the order of subset
            pin_memory=True,
        )

        arg = get_kcg(model, len(all_data.annot_data), unlabeled_loader, addendum, subset)

    if method == "lloss":
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(
            all_data, batch_size=BATCH, sampler=SubsetSequentialSampler(subset), pin_memory=True
        )

        # Measure uncertainty of each data points in the subset
        uncertainty = get_uncertainty(model, unlabeled_loader)
        arg = np.argsort(uncertainty)

    if method == "VAAL":
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(
            all_data, batch_size=BATCH, sampler=SubsetSequentialSampler(subset), pin_memory=True
        )
        labeled_loader = DataLoader(
            all_data, batch_size=BATCH, sampler=SubsetSequentialSampler(labeled_set), pin_memory=True
        )
        if data_type in ["mnist", "fashion_mnist"]:
            vae = VAE(28, 1, 3)
            discriminator = Discriminator(28)
        elif data_type == "tata_small":
            vae = VAE(32, 1, 4)
            discriminator = Discriminator(32)
        elif data_type in ["svhn", "cifar10"]:
            vae = VAE(32, 3, 4)
            discriminator = Discriminator(32)
        else:
            raise ValueError("`data_type` {} not supported by VAAL.".format(data_type))
        models = {"vae": vae, "discriminator": discriminator}

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        optimizers = {"vae": optim_vae, "discriminator": optim_discriminator}

        train_vaal(models, optimizers, labeled_loader, unlabeled_loader, subset)

        all_preds, all_indices = [], []

        for images, _, indices in unlabeled_loader:
            images = images.cuda()
            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1
        # select the points which the discriminator things are the most likely to be unlabeled
        _, arg = torch.sort(all_preds)

    return arg


class GCNDataWrapper(Dataset):
    """
    This Dataset class translates between the data structures from the main repo and the GCN data structures.
    """

    def __init__(self, annot_data, pool_data):
        # `annot_data` (/ `pool_data) is a DataWrapper object from npal.data.data_module.
        self.annot_data = annot_data
        self.pool_data = pool_data

        # Numpy arrays
        aX, ay = annot_data.get_data()  # Already normalised
        pX, py = pool_data.get_data()
        # To tensor
        self.aX = torch.from_numpy(aX).float()
        self.ay = torch.from_numpy(ay).float()
        self.pX = torch.from_numpy(pX).float()
        self.py = torch.from_numpy(py).float()

    def __getitem__(self, index):
        # Annot data has lowest indices. If index geq than annot length, get from pool instead.
        if index >= len(self.annot_data):
            data = self.pX[index - len(self.annot_data), ...]
            target = self.py[index - len(self.annot_data)]
        else:
            data = self.aX[index, ...]
            target = self.ay[index]

        return data, target, index

    def __len__(self):
        return len(self.annot_data) + len(self.pool_data)


class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
