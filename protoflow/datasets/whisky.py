"""Whisky dataset for classification."""
import numpy as np
from protoflow.utils.data import get_file_from_google


def load_data(path="whisky.npz"):
    """Loads the Whisky dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets/).

    # Returns
        Tuple of Numpy arrays: `x_all, y_all`.
    """
    path = get_file_from_google(
        path,
        file_id="1Q70buiE5GGmprHcHFCs0ZTxuLaAqJW8U",
        md5_hash="ba0bbf37cc81219d00d8f2af19b89dfd",
        extract=False,
    )
    with np.load(path, allow_pickle=False) as f:
        x_all, y_all = f["x_all"], f["y_all"]
    return (x_all, y_all)
