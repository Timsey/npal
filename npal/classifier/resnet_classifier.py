"""
Implementation adapted from: https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning
"""


import logging
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from npal.utils import count_trainable_parameters
from npal.classifier.base_classifier import BaseClassifier
from npal.acquisition.gcn_architecture.gcn_resnet import ResNet18
from npal.acquisition.gcn_architecture.gcn_models import LossNet, loss_pred_loss


# Original values:
# MILESTONES = [160, 240]
#
# BATCH = 128  # B
# EPOCH = 200
# EPOCHL = 120  # 20 #120 # After this many epochs, stop
#
# MARGIN = 1.0  # xi
# WEIGHT = 1.0  # lambda
#
# LR = 1e-1
# MOMENTUM = 0.9
# WDECAY = 5e-4  # 2e-3# 5e-4


# For 100 points start, budget 100? Unnecessary.
# MILESTONES = [80, 120]
# EPOCH = 100  # 200
# EPOCHL = 60
# BATCH = 128

# Defaults
MILESTONES = [160, 240]
EPOCH = 200  # 200
EPOCHL = 120
BATCH = 128

MARGIN = 1.0  # xi
WEIGHT = 1.0  # lambda

LR = 1e-1
MOMENTUM = 0.9
WDECAY = 5e-4


# For reproducibility of classifier
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Numpy2TorchDataset(Dataset):
    """Wrapper between numpy and torch datasets."""

    def __init__(self, X, y=None):
        super().__init__()
        self.X = torch.from_numpy(X)
        if y is not None:
            self.y = torch.from_numpy(y)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx, ...]
        else:
            return self.X[idx, ...], self.y[idx]


class ResNetClassifier(BaseClassifier):
    def __init__(self, args, classes, remaining_classes, imbalance_factors, suppress_train_log=False):
        super().__init__(args, classes, remaining_classes, imbalance_factors)
        # Reset torch seeds every time a classifier is created, for consistency between computed improvements and
        #  actual improvements.
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        self.use_class_weights_for_fit = args.use_class_weights_for_fit

        self.logger = logging.getLogger("ResNetClassifier")
        self.report_interval = EPOCH // 10
        self.suppress_train_log = suppress_train_log

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.learning_loss = args.acquisition_strategy == "gcn_lloss"
        if args.data_type in ["mnist", "fashion_mnist"]:
            self.model = ResNetWrapper(
                input_shape=(28, 28),
                num_channels=1,
                num_classes=len(remaining_classes),
                learning_loss=self.learning_loss,
            ).to(self.device)
        elif args.data_type in ["svhn", "cifar10"]:
            self.model = ResNetWrapper(
                input_shape=(32, 32),
                num_channels=3,
                num_classes=len(remaining_classes),
                learning_loss=self.learning_loss,
            ).to(self.device)
        else:
            raise NotImplementedError("No ResNet implemented for data type {}.".format(args.data_type))

        if not suppress_train_log:
            self.logger.info("ResNet with {} params.".format(count_trainable_parameters(self.model.resnet18)))
            if self.learning_loss:
                self.logger.info(" + LossNet with {} params.".format(count_trainable_parameters(self.model.lloss)))

        if self.use_class_weights_for_fit:
            self.loss_func = nn.CrossEntropyLoss(
                weight=torch.from_numpy(self.class_weights[: len(remaining_classes)]).float().to(self.device),
                reduction="none",
            )  # accepts non one-hot class labels as targets
        else:
            self.loss_func = nn.CrossEntropyLoss(reduction="none")  # accepts non one-hot class labels as targets
        # self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)
        self.optim = torch.optim.SGD(self.model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=MILESTONES)

    def fit(self, X, y):
        train_dataset = Numpy2TorchDataset(X, y)
        # Define DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH,
            shuffle=True,
            worker_init_fn=seed_worker,
        )
        self.model.train()
        self.loss_func.train()
        for epoch in range(EPOCH):
            train_loss = self.model.train_epoch(
                train_loader, self.device, self.loss_func, self.optim, self.scheduler, epoch
            )

            if not self.suppress_train_log and epoch % self.report_interval == 0:
                log_str = "{:>7}, {:>19}".format(
                    f"Epoch {epoch:>2}",
                    f"train loss: {train_loss:>5.3f}",
                )
                self.logger.info(log_str)

    def predict(self, X):
        self.model.eval()
        self.loss_func.eval()
        dataset = Numpy2TorchDataset(X, y=None)
        # Define DataLoader
        loader = DataLoader(
            dataset,
            batch_size=1024,
            shuffle=False,
        )
        all_preds = []
        with torch.no_grad():
            for X in loader:
                X = X.to(self.device)
                logits, _, _ = self.model.forward(X)
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()  # Class labels
                all_preds.append(preds)
        return np.concatenate(all_preds, axis=0)

    def predict_proba(self, X):
        self.model.eval()
        self.loss_func.eval()
        dataset = Numpy2TorchDataset(X, y=None)
        # Define DataLoader
        loader = DataLoader(
            dataset,
            batch_size=1024,
            shuffle=False,
        )
        all_probs = []
        with torch.no_grad():
            for X in loader:
                X = X.to(self.device)
                logits, _, _ = self.model.forward(X)
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    def get_num_classifier_features(self):
        return len(self.remaining_classes)  # number of probabilities equal to number of remaining classes

    def get_classifier_features(self, X):
        return self.predict_proba(X)  # (n_samples x n_remaining_classes)

    def get_classifier_embedding(self, X):
        """Used in some hybrid strategies. """
        self.model.eval()
        self.loss_func.eval()
        dataset = Numpy2TorchDataset(X, y=None)
        # Define DataLoader
        loader = DataLoader(
            dataset,
            batch_size=1024,
            shuffle=False,
        )
        all_embeddings = []
        with torch.no_grad():
            for X in loader:
                X = X.to(self.device)
                _, flat_pre_logits, _ = self.model.forward(X)
                embedding = flat_pre_logits.detach().cpu().numpy()
                all_embeddings.append(embedding)
        return np.concatenate(all_embeddings, axis=0)


class ResNetWrapper(nn.Module):
    def __init__(self, input_shape, num_channels, num_classes, learning_loss=False):
        super().__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_loss = learning_loss

        self.image_channels = num_channels
        grayscale = num_channels == 1
        if input_shape == (28, 28):  # MNIST, FashionMNIST
            feature_sizes = [28, 14, 7, 4]
        elif input_shape == (32, 32):  # SVHN, CIFAR10
            feature_sizes = [32, 16, 8, 4]
        else:
            raise ValueError("Expected `input_shape` to be (28, 28) or (32, 32), not {}.".format(input_shape))

        self.resnet18 = ResNet18(num_classes=num_classes, grayscale=grayscale)

        self.lloss = None
        if learning_loss:
            self.lloss = LossNet(feature_sizes=feature_sizes, num_channels=[64, 128, 256, 512], interm_dim=128)
            self.lloss_loss_func = loss_pred_loss
            self.lloss_optim = torch.optim.SGD(self.lloss.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
            self.lloss_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.lloss_optim, milestones=MILESTONES)

    def forward(self, x):
        # First reshape image to (B, C, H, W) if necessary
        b = x.shape[0]
        x = x.reshape((b, self.image_channels, self.input_shape[0], self.input_shape[1]))
        x = x.float()

        # ResNet forward
        logits, flat_pre_logits, features = self.resnet18(x)
        return logits, flat_pre_logits, features

    def train_epoch(self, loader, device, loss_func, optim, scheduler, epoch):
        epoch_loss = 0
        num_examples = 0
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            num_examples += y.shape[0]
            optim.zero_grad()
            if self.learning_loss:
                self.lloss_optim.zero_grad()
            logits, _, features = self.forward(X)
            target_loss = loss_func(logits, y)
            if self.learning_loss:
                if epoch > EPOCHL:
                    features[0] = features[0].detach()
                    features[1] = features[1].detach()
                    features[2] = features[2].detach()
                    features[3] = features[3].detach()
                pred_loss = self.lloss(features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                m_module_loss = self.lloss_loss_func(pred_loss, target_loss, margin=MARGIN)
                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                epoch_loss += torch.sum(target_loss) + m_module_loss * target_loss.size(0)
                loss = m_backbone_loss + WEIGHT * m_module_loss
            else:
                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                epoch_loss += torch.sum(target_loss)
                loss = m_backbone_loss
            loss.backward()
            optim.step()
            if self.learning_loss:
                self.lloss_optim.step()
        epoch_loss /= num_examples

        scheduler.step()
        if self.learning_loss:
            self.lloss_scheduler.step()
        return epoch_loss

    # def val_epoch(self, loader, device):
    #     epoch_loss = 0
    #     num_examples = 0
    #     for X, y in loader:
    #         X = X.to(device)
    #         y = y.to(device)
    #         num_examples += y.shape[0]
    #         logits = self.forward(X)
    #         sum_loss = self.loss_func(logits, y)
    #         epoch_loss += sum_loss
    #     epoch_loss /= num_examples
    #     return epoch_loss
