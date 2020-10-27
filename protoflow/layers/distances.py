"""ProtoFlow Distance layers."""

import tensorflow as tf

import protoflow as pf


class _Distance(tf.keras.layers.Layer):
    """Base layer for all ProtoFlow Distance layers.

    Assume that the last `nprotos` of the inputs are the prototypes
    and compute the pairwise distance between the data and the prototypes
    along with the `prototype_labels`.

    Arguments:
        prototype_labels: (list), class labels of the prototypes.
        forward_labels: (bool), whether or not to append the prototypes_labels
            to the output.
    """
    def __init__(self, prototype_labels, forward_labels=True, **kwargs):
        super().__init__(**kwargs)
        self.prototype_labels = tf.constant(prototype_labels, dtype=self.dtype)
        self.forward_labels = forward_labels

    def distance(self, x, w):
        raise NotImplementedError("Distance method not implemented.")

    def call(self, inputs):
        nprotos = len(self.prototype_labels)
        x = inputs[:-nprotos]
        w = inputs[-nprotos:]
        d = self.distance(x, w)
        plabels = tf.reshape(self.prototype_labels, shape=(1, -1))
        if self.forward_labels:
            return tf.concat([d, plabels],
                             axis=0,
                             name="append_prototype_labels")
        return d

    # def compute_output_shape(self, input_shape):
    #     nrows = input_shape[0] + len(self.prototype_labels)
    #     ncols = input_shape[1]
    #     if self.forward_labels:
    #         nrows += 1
    #     return (nrows, ncols)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "prototype_labels": self.prototype_labels.numpy(),
            "forward_labels": self.forward_labels,
        }
        return {**base_config, **config}


class Euclidean(_Distance):
    """Thin wrapper layer for the Euclidean Distance."""
    def distance(self, x, w):
        d = pf.functions.distances.euclidean_distance(x, w)
        return d


class SquaredEuclidean(_Distance):
    """Thin wrapper layer for the Squared-Euclidean Distance."""
    def distance(self, x, w):
        d = pf.functions.distances.squared_euclidean_distance(x, w)
        return d
