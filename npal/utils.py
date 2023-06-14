import pickle
import wandb
import numpy as np
import matplotlib.pyplot as plt


def save_object(obj, filename):
    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(obj, output, -1)


def load_object(filename):
    with open(filename, "rb") as input:
        data = pickle.load(input)
    return data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) if model is not None else 0


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) if model is not None else 0


def count_untrainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad) if model is not None else 0


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def str2none(v):
    if v is None:
        return v
    if v.lower() == "none":
        return None
    else:
        return v


def str2none_int(v):
    if v is None:
        return v
    if v.lower() == "none":
        return None
    else:
        return int(v)


def str2none_float(v):
    if v is None:
        return v
    if v.lower() == "none":
        return None
    else:
        return float(v)


def log_hist_of_labels(labels, step):
    def bins_labels(bins, **kwargs):
        bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
        plt.xticks(np.arange(min(bins) + bin_w / 2, max(bins), bin_w), bins[:-1], **kwargs)
        plt.xlim(bins[0], bins[-1])

    plt.figure(figsize=(8, 5))
    bins = range(len(set(labels)) + 1)
    plt.hist(labels, bins=bins)
    plt.title("Current labels", fontsize=18)
    plt.ylabel("Frequency", fontsize=15)
    plt.xlabel("Class", fontsize=15)
    # Customise class labels to be halfway the ticks
    bins_labels(bins)
    plt.tight_layout()
    wandb.log({f"label_dist{step}": wandb.Image(plt)})
    plt.close()
