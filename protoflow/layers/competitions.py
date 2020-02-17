"""ProtoFlow Competition layers."""

import tensorflow as tf


class WTAC(tf.keras.layers.Layer):
    """Winner-Takes-All Competition.

    Arguments:
        prototype_labels: (list), class labels of the prototypes.
    """
    def __init__(self, prototype_labels, **kwargs):
        self.prototype_labels = prototype_labels
        super().__init__(**kwargs)

    def call(self, x):
        neg_distances = tf.negative(x)
        _, winning_indices = tf.nn.top_k(neg_distances, k=1)
        winning_labels = tf.gather(self.prototype_labels, winning_indices)
        y = tf.squeeze(winning_labels)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(WTAC, self).get_config()
        config = {
            'prototype_labels': self.prototype_labels,
        }
        return {**base_config, **config}


class KNNC(tf.keras.layers.Layer):
    """K-Nearest-Neighbors Competition.

    Arguments:
        oh_prototype_labels: (list), one-hot class labels of the prototypes.
    """
    def __init__(self, k, oh_prototype_labels, **kwargs):
        self.k = k

        # One-hot prototype labels
        self.oh_prototype_labels = oh_prototype_labels

        super().__init__(**kwargs)

    def call(self, x):
        neg_distances = tf.negative(x)
        _, winning_indices = tf.nn.top_k(neg_distances, k=self.k)
        top_k_labels = tf.gather(self.oh_prototype_labels, winning_indices)
        predictions_sum = tf.reduce_sum(input_tensor=top_k_labels, axis=1)
        y_pred = tf.argmax(input=predictions_sum, axis=1)
        return y_pred

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(KNNC, self).get_config()
        config = {
            'k': self.k,
            'oh_prototype_labels': self.oh_prototype_labels,
        }
        return {**base_config, **config}
