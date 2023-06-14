"""
Copied and adapted from: https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning
"""

import abc
from sklearn.metrics import pairwise_distances

import numpy as np
import torch
import torch.nn as nn


CUDA_VISIBLE_DEVICES = 0

BATCH = 128  # B
# SUBSET = 10000  # M
# ADDENDUM = 1000  # K
EPOCHV = 100  # VAAL number of epochs


def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl)
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj * unlabeled_score
    return bce_adj_loss


def aff_to_adj(x, y=None):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj += -1.0 * np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0)  # rowise sum
    adj = np.matmul(adj, np.diag(1 / adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()
    return adj


def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD


def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label, _ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img


def get_uncertainty(models, unlabeled_loader):
    models["backbone"].eval()
    models["module"].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            _, _, features = models["backbone"](inputs)
            pred_loss = models["module"](features)  # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))
            uncertainty = torch.cat((uncertainty, pred_loss), 0)

    return uncertainty.cpu()


def get_features(models, unlabeled_loader):
    models["backbone"].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()
    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                _, features_batch, _ = models["backbone"](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features  # .detach().cpu().numpy()
    return feat


def get_kcg(models, labeled_data_size, unlabeled_loader, addendum, subset):
    models["backbone"].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            _, features_batch, _ = models["backbone"](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        new_av_idx = np.arange(len(subset), (len(subset) + labeled_data_size))
        sampling = kCenterGreedy(feat)
        batch = sampling.select_batch_(new_av_idx, addendum)
        other_idx = [x for x in range(len(subset)) if x not in batch]
    return other_idx + batch


def train_vaal(models, optimizers, labeled_dataloader, unlabeled_dataloader, subset):

    vae = models["vae"]
    discriminator = models["discriminator"]
    vae.train()
    discriminator.train()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        vae = vae.cuda()
        discriminator = discriminator.cuda()

    adversary_param = 1
    beta = 1
    num_adv_steps = 1
    num_vae_steps = 2

    bce_loss = nn.BCELoss()

    labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)
    # labeled_dataloader.dataset is the GCNDataWrapper object that contains `annot_data` and `pool_data` attributes.
    train_iterations = int((len(subset) + len(labeled_dataloader.dataset.annot_data)) * EPOCHV / BATCH)

    for iter_count in range(train_iterations):
        labeled_imgs, labels = next(labeled_data)
        unlabeled_imgs = next(unlabeled_data)[0]

        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            labeled_imgs = labeled_imgs.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()
            labels = labels.cuda()

        # VAE step
        for count in range(num_vae_steps):  # num_vae_steps
            recon, _, mu, logvar = vae(labeled_imgs)
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            unlab_recon, _, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, unlab_recon, unlab_mu, unlab_logvar, beta)

            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)

            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))

            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_real_preds = unlab_real_preds.cuda()

            dsc_loss = bce_loss(labeled_preds[:, 0], lab_real_preds) + bce_loss(unlabeled_preds[:, 0], unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss

            optimizers["vae"].zero_grad()
            total_vae_loss.backward()
            optimizers["vae"].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_vae_steps - 1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(labeled_imgs)
                _, _, unlab_mu, _ = vae(unlabeled_imgs)

            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)

            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_fake_preds = unlab_fake_preds.cuda()

            dsc_loss = bce_loss(labeled_preds[:, 0], lab_real_preds) + bce_loss(unlabeled_preds[:, 0], unlab_fake_preds)

            optimizers["discriminator"].zero_grad()
            dsc_loss.backward()
            optimizers["discriminator"].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_adv_steps - 1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()
            if iter_count % 100 == 0:
                print(
                    "Iteration: "
                    + str(iter_count)
                    + "  vae_loss: "
                    + str(total_vae_loss.item())
                    + " dsc_loss: "
                    + str(dsc_loss.item())
                )


class SamplingMethod(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, X, y, seed, **kwargs):
        self.X = X
        self.y = y
        self.seed = seed

    def flatten_X(self):
        shape = self.X.shape
        flat_X = self.X
        if len(shape) > 2:
            flat_X = np.reshape(self.X, (shape[0], np.product(shape[1:])))
        return flat_X

    @abc.abstractmethod
    def select_batch_(self):
        return

    def select_batch(self, **kwargs):
        return self.select_batch_(**kwargs)

    def select_batch_unc_(self, **kwargs):
        return self.select_batch_unc_(**kwargs)

    def to_dict(self):
        return None


class kCenterGreedy(SamplingMethod):
    def __init__(self, X, metric="euclidean"):
        self.X = X
        # self.y = y
        self.flat_X = self.flatten_X()
        self.name = "kcenter"
        self.features = self.flat_X
        self.metric = metric
        self.min_distances = None
        self.max_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers if d not in self.already_selected]
        if cluster_centers:
            x = self.features[cluster_centers]
            # Update min_distances for all examples given new cluster center.
            dist = pairwise_distances(self.features, x, metric=self.metric)  # ,n_jobs=4)

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, already_selected, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        try:
            # Assumes that the transform function takes in original data and not
            # flattened data.
            print("Getting transformed features...")
            #   self.features = model.transform(self.X)
            print("Calculating distances...")
            self.update_distances(already_selected, only_new=False, reset_dist=True)
        except:
            print("Using flat_X as features.")
            self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []

        for _ in range(N):
            if self.already_selected is None:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        print("Maximum distance from cluster centers is %0.2f" % max(self.min_distances))

        self.already_selected = already_selected

        return new_batch
