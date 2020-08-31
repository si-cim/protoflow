"""Module to build and evaluate KNN models with Keras."""

import numpy as np
import tensorflow as tf

from protoflow import layers
from protoflow.applications.network import Network


class KNN(Network):
    """K-Nearest Neighbors (KNN)."""
    def __init__(self, k=3, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def build(self, x_train, y_train):
        self._check_validity(x_train, y_train)

        self.num_classes = np.unique(y_train).size
        self.prototypes, self.prototype_labels = x_train, y_train

        # One-hot encoding
        y_train = np.eye(self.num_classes)[y_train]

        # Define layers
        self.distance = layers.SquaredEuclideanDistance(
            num_of_prototypes=x_train.shape[0],
            input_dim=x_train.shape[1],
            prototype_labels=y_train,
            name='eucl_dist')
        self.knnc = layers.KNNC(k=self.k, oh_prototype_labels=y_train)

        # Build model
        self.model = tf.keras.models.Sequential([self.distance, self.knnc])

        self.built = True

    def fit(self, x_train, y_train, **kwargs):
        """Fit only saves the training data.

        KNN does not "learn". It is only a pseudo learner.

        Arguments:
            x_train : Training inputs.
            y_train : Training targets.
        """
        if not self.built:
            self.build(x_train, y_train)
        weights = self.distance.get_weights()
        weights[-1] = x_train
        self.distance.set_weights(weights)
