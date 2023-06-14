import pickle
import numpy as np

"""
Dataset info:

(8124, 22)  mushrooms.p
(5000, 21)  waveform.p
(1605, 119) adult.p
"""

UCI_DATA_NAMES = [
    "adult",
    "mushrooms",
    "waveform",
]


def load_uci_data(args):
    # Load data from disk
    data_path = args.abs_path_to_data_store / "dataUCI" / f"{args.data_type}.p"
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        X = data["X"]
        # Some datasets have float labels: adult
        y = data["y"][:, 0].astype(np.int32)  # Make (num_data, ) instead of (num_data, 1), convert float labels to int

    # Some UCI datasets are matrices instead of arrays.
    return np.asarray(X), np.asarray(y), None, None  # train_X, train_y, test_X, test_y
