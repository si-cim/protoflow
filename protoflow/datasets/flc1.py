"""FLC1 dataset for classification."""
import numpy as np
from tensorflow.keras.utils import get_file


def load_data(path='flc1.npz'):
    """Loads the Whisky dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets/).

    # Returns
        Tuple of Numpy arrays: `x_all, y_all`.
    """
    path = get_file(path, origin='http://tiny.cc/pfds_flc1')
    with np.load(path, allow_pickle=False) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)
