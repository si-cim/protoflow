"""ProtoFlow loss functions."""

import six
import tensorflow as tf
from tensorflow.keras.regularizers import *  # noqa: F403, F401
from tensorflow.keras.utils import (deserialize_keras_object,
                                    serialize_keras_object)

from protoflow.functions import activations


class GLVQLoss(tf.keras.losses.Loss):
    """Loss function based on Prototypes."""
    def __init__(self,
                 prototype_labels,
                 squashing='beta_sigmoid',
                 beta=20,
                 minimize_dp=False,
                 **kwargs):
        self.prototype_labels = prototype_labels
        self.squashing = activations.get(squashing)
        self.beta = beta
        self.minimize_dp = minimize_dp
        super().__init__(**kwargs)

    def call(self, y_true, distances):
        y_true = tf.cast(y_true, distances.dtype)
        matching = tf.equal(y_true, self.prototype_labels)
        unmatching = tf.logical_not(matching)

        inf = tf.constant(float('inf'))
        d_matching = tf.where(matching, distances, inf)
        d_unmatching = tf.where(unmatching, distances, inf)
        dp = tf.keras.backend.min(d_matching, axis=1, keepdims=True)
        dm = tf.keras.backend.min(d_unmatching, axis=1, keepdims=True)

        mu = (dp - dm) / (dp + dm)

        batch_loss = self.squashing(mu, beta=self.beta)

        reg_term = 0
        if self.minimize_dp:
            reg_term = dp

        return batch_loss + reg_term

    def get_config(self):
        base_config = super().get_config()
        config = {
            'prototype_labels': self.prototype_labels,
            'squashing': activations.serialize(self.squashing),
            'beta': self.beta,
        }
        # Concatenate the dictionaries
        return {**base_config, **config}


def serialize(loss):
    return serialize_keras_object(loss)


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='ProtoFlow loss')


def get(identifier):
    """Get the `identifier` loss function.

    # Arguments
        identifier: None or str, name of the function.

    # Returns
        The loss function or None if `identifier` is None.

    # Raises
        ValueError if unknown identifier.
    """
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'loss function identifier:', identifier)
