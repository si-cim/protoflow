"""Whisky dataset for classification."""
import numpy as np
from tensorflow.keras.utils import get_file


def load_data(path='whisky.npz'):
    """Loads the Whisky dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets/).

    # Returns
        Tuple of Numpy arrays: `x_all, y_all`.
    """
    path = get_file(path, origin='http://tiny.cc/pfds_whisky')
    with np.load(path, allow_pickle=False) as f:
        x_all, y_all = f['x_all'], f['y_all']
    return (x_all, y_all)
