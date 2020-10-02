"""Build ProtoFlow models with tf.keras."""

import tensorflow as tf


class Network(tf.keras.models.Model):
    """Abstract Class for ProtoFlow Applications.

    Keyword Arguments:
        TODO (str) : To be filled in later.
    """
    def __init__(self, **kwargs):
        super().__init__(kwargs)
