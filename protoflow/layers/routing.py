"""ProtoFlow routing layers."""

import tensorflow as tf

import protoflow as pf


class ExpandDims(tf.keras.layers.Layer):
    """Thin wrapper for `tf.expand_dims`. Because Keras complains
    when trainable prototypes are routed through a lambda layer.

    Arguments:
        axis: (int), the axis to expand.
    """
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def call(self, x):
        return tf.expand_dims(x, axis=self.axis)


class StratifiedMinimum(tf.keras.layers.Layer):
    """Return the minimum in each category for every sample.

    Arguments:
        category_labels: (list), labels of the categories.
    """
    def __init__(self, prototype_labels, **kwargs):
        super().__init__(**kwargs)
        self.prototype_labels = tf.constant(prototype_labels, dtype=self.dtype)

    def call(self, x):
        # return pf.functions.stratified_min(x, self.prototype_labels)
        raise NotImplementedError("TODO")
