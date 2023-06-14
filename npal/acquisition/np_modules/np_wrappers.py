import torch

from .npf.neuralproc import CNP
from .npf import AttnCNP
from .npf.architectures import MLP


class MergeFeatureCNP(CNP):
    """Wrapper for CNP to use extra input features in X_cntxt and X_trgt.

    This module checks if there are extra features beyond the data features (e.g. classifier features), and if so,
    it first combines the two types of features, by resizing the extra feature dim to the data feature dim with an
    MLP. This module also contains functionality for projecting both features to a higher dimensional space before
    combining. If so, the base NP will take as input tensors of this higher dimensional size (r_dim).
    """

    def __init__(self, x_dim, y_dim, r_dim, num_extra_features, project_np_features, XYEncoder, **kwargs):
        # Init base NP
        if project_np_features:
            super().__init__(r_dim, y_dim, XYEncoder, r_dim=r_dim, **kwargs)
        else:
            super().__init__(x_dim, y_dim, XYEncoder, r_dim=r_dim, **kwargs)

        self.num_extra_features = num_extra_features
        self.project_np_features = project_np_features
        assert num_extra_features >= 0, "Cannot have negative number of extra features."
        if project_np_features:  # Project data features to r_dim
            self.cntxt_resizer = MLP(x_dim, r_dim, hidden_size=r_dim)
            self.trgt_resizer = MLP(x_dim, r_dim, hidden_size=r_dim)
            if num_extra_features > 0:  # Project extra features to r_dim
                self.cntxt_resizer_extra = MLP(num_extra_features, r_dim, hidden_size=r_dim)
                self.trgt_resizer_extra = MLP(num_extra_features, r_dim, hidden_size=r_dim)
            else:
                pass  # Just the data features projection is required
        else:  # No projection
            if num_extra_features > 0:  # resize extra features to combine with data features: project to x_dim
                self.cntxt_resizer_extra = MLP(num_extra_features, x_dim, hidden_size=r_dim)
                self.trgt_resizer_extra = MLP(num_extra_features, x_dim, hidden_size=r_dim)
            else:
                pass  # Just pass features to standard NP

    def forward(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
        # NOTE: Using tanh as non-linearity here because NP needs inputs in [-1, 1]
        if self.project_np_features:
            if self.num_extra_features > 0:
                # Project data and extra features, sum, then tanh
                X_cntxt = torch.tanh(
                    self.cntxt_resizer(X_cntxt[..., : -self.num_extra_features])
                    + self.cntxt_resizer_extra(X_cntxt[..., -self.num_extra_features :])
                )
                X_trgt = torch.tanh(
                    self.trgt_resizer(X_trgt[..., : -self.num_extra_features])
                    + self.trgt_resizer_extra(X_trgt[..., -self.num_extra_features :])
                )
            else:
                # Project data features, then tanh
                X_cntxt = torch.tanh(self.cntxt_resizer(X_cntxt))
                X_trgt = torch.tanh(self.trgt_resizer(X_trgt))
        else:
            if self.num_extra_features > 0:
                # Resize extra features, sum with data features, then tanh
                X_cntxt = torch.tanh(
                    X_cntxt[..., : -self.num_extra_features]
                    + self.cntxt_resizer_extra(X_cntxt[..., -self.num_extra_features :])
                )
                X_trgt = torch.tanh(
                    X_trgt[..., : -self.num_extra_features]
                    + self.trgt_resizer_extra(X_trgt[..., -self.num_extra_features :])
                )
            else:
                # Don't project data features, and no extra features to use
                pass

        # Forward pass of the base NP
        return super().forward(X_cntxt, Y_cntxt, X_trgt, Y_trgt)


class MergeFeatureAttnCNP(AttnCNP):
    # TODO: This is pretty much a copy of MergeFeatureCNP, but with a different superclass (necessary for super()).
    #  Maybe this can be refactored?
    """Wrapper for AttnCNP to use extra input features in X_cntxt and X_trgt.

    This module checks if there are extra features beyond the data features (e.g. classifier features), and if so,
    it first combines the two types of features, by resizing the extra feature dim to the data feature dim with an
    MLP. This module also contains functionality for projecting both features to a higher dimensional space before
    combining. If so, the base NP will take as input tensors of this higher dimensional size (r_dim).
    """

    def __init__(self, x_dim, y_dim, r_dim, num_extra_features, project_np_features, XYEncoder, **kwargs):
        # Init base NP
        if project_np_features:
            super().__init__(r_dim, y_dim, XYEncoder, r_dim=r_dim, **kwargs)
        else:
            super().__init__(x_dim, y_dim, XYEncoder, r_dim=r_dim, **kwargs)

        self.num_extra_features = num_extra_features
        self.project_np_features = project_np_features
        assert num_extra_features >= 0, "Cannot have negative number of extra features."
        if project_np_features:  # Project data features to r_dim
            self.cntxt_resizer = MLP(x_dim, r_dim, hidden_size=r_dim)
            self.trgt_resizer = MLP(x_dim, r_dim, hidden_size=r_dim)
            if num_extra_features > 0:  # Project extra features to r_dim
                self.cntxt_resizer_extra = MLP(num_extra_features, r_dim, hidden_size=r_dim)
                self.trgt_resizer_extra = MLP(num_extra_features, r_dim, hidden_size=r_dim)
        else:  # No projection
            if num_extra_features > 0:  # resize extra features to combine with data features: project to x_dim
                self.cntxt_resizer_extra = MLP(num_extra_features, x_dim, hidden_size=r_dim)
                self.trgt_resizer_extra = MLP(num_extra_features, x_dim, hidden_size=r_dim)
            else:
                pass  # Just pass features to standard NP

    def forward(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
        # NOTE: Using tanh as non-linearity here because NP needs inputs in [-1, 1]
        if self.project_np_features:
            if self.num_extra_features > 0:
                # Project data and extra features, sum, then tanh
                X_cntxt = torch.tanh(
                    self.cntxt_resizer(X_cntxt[..., : -self.num_extra_features])
                    + self.cntxt_resizer_extra(X_cntxt[..., -self.num_extra_features :])
                )
                X_trgt = torch.tanh(
                    self.trgt_resizer(X_trgt[..., : -self.num_extra_features])
                    + self.trgt_resizer_extra(X_trgt[..., -self.num_extra_features :])
                )
            else:
                # Project data features, then tanh
                X_cntxt = torch.tanh(self.cntxt_resizer(X_cntxt))
                X_trgt = torch.tanh(self.trgt_resizer(X_trgt))
        else:
            if self.num_extra_features > 0:
                # Resize extra features, sum with data features, then tanh
                X_cntxt = torch.tanh(
                    X_cntxt[..., : -self.num_extra_features]
                    + self.cntxt_resizer_extra(X_cntxt[..., -self.num_extra_features :])
                )
                X_trgt = torch.tanh(
                    X_trgt[..., : -self.num_extra_features]
                    + self.trgt_resizer_extra(X_trgt[..., -self.num_extra_features :])
                )
            else:
                # Don't project data features, and no extra features to use
                pass

        # Forward pass of the base NP
        return super().forward(X_cntxt, Y_cntxt, X_trgt, Y_trgt)
