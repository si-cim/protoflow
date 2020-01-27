"""Tecator dataset for classification.

URL:
    http://lib.stat.cmu.edu/datasets/tecator

LICENCE / TERMS / COPYRIGHT:
    This is the Tecator data set: The task is to predict the fat content
    of a meat sample on the basis of its near infrared absorbance spectrum.
    -------------------------------------------------------------------------
    1. Statement of permission from Tecator (the original data source)

    These data are recorded on a Tecator Infratec Food and Feed Analyzer
    working in the wavelength range 850 - 1050 nm by the Near Infrared
    Transmission (NIT) principle. Each sample contains finely chopped pure
    meat with different moisture, fat and protein contents.

    If results from these data are used in a publication we want you to
    mention the instrument and company name (Tecator) in the publication.
    In addition, please send a preprint of your article to

        Karin Thente, Tecator AB,
        Box 70, S-263 21 Hoganas, Sweden

    The data are available in the public domain with no responsability from
    the original data source. The data can be redistributed as long as this
    permission note is attached.

    For more information about the instrument - call Perstorp Analytical's
    representative in your area.

Description:
    For each meat sample the data consists of a 100 channel spectrum of
    absorbances and the contents of moisture (water), fat and protein.
    The absorbance is -log10 of the transmittance
    measured by the spectrometer. The three contents, measured in percent,
    are determined by analytic chemistry.
"""

import numpy as np
from tensorflow.keras.utils import get_file


def load_data(fname='tecator.npz'):
    """Loads the Tecator dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `x_train, y_train`.
    """
    path = get_file(fname,
                    origin='http://tiny.cc/anysma_datasets_tecator',
                    file_hash='7b12e6d01131abc468224907e8718fa0')
    with np.load(path, allow_pickle=False) as f:
        x_train, y_train = f['x_train'], f['y_train']
    return x_train, y_train
