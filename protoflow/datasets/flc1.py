"""FLC1 dataset for classification."""
import numpy as np

from protoflow.utils.data import get_file_from_google


def load_data(path='flc1.npz'):
    """Loads the Whisky dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets/).

    # Returns
        Tuple of Numpy arrays: `x_all, y_all`.
    """
    path = get_file_from_google(
        path,
        file_id="1BF3_HT8-jmSQlPBOR1Xk_D73a5W2JkRF",
        md5_hash="b07b230d923b3a75fba3c039d2429a24",
        extract=False,
    )
    with np.load(path, allow_pickle=False) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)
