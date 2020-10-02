"""ProtoFlow Competition layers."""

import tensorflow as tf

import protoflow as pf


class WTAC(tf.keras.layers.Layer):
    """Winner-Takes-All Competition.

    Arguments:
        prototype_labels: (list), class labels of the prototypes.
    """
    def __init__(self, prototype_labels, **kwargs):
        self.prototype_labels = prototype_labels
        super().__init__(**kwargs)

    def call(self, distances):
        y_pred = pf.functions.wtac(distances, self.prototype_labels)
        return y_pred

    def compute_output_shape(self, input_shape):
        return (input_shape[0], )

    def get_config(self):
        base_config = super(WTAC, self).get_config()
        config = {
            'prototype_labels': self.prototype_labels,
        }
        return {**base_config, **config}


class KNNC(tf.keras.layers.Layer):
    """K-Nearest-Neighbors Competition.

    Arguments:
        one_hot_prototype_labels: (list), one-hot class labels
            of the prototypes.
    """
    def __init__(self, k, one_hot_prototype_labels, **kwargs):
        self.k = k
        self.one_hot_prototype_labels = one_hot_prototype_labels
        super().__init__(**kwargs)

    def call(self, distances):
        y_pred = pf.functions.knnc(distances,
                                   self.one_hot_prototype_labels,
                                   k=self.k)
        return y_pred

    def compute_output_shape(self, input_shape):
        return (input_shape[0], )

    def get_config(self):
        base_config = super(KNNC, self).get_config()
        config = {
            'k': self.k,
            'oh_prototype_labels': self.oh_prototype_labels,
        }
        return {**base_config, **config}
