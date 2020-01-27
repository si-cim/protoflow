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
                 prototype_labels=None,
                 squashing='beta_sigmoid',
                 beta=20,
                 **kwargs):
        self.prototype_labels = prototype_labels
        self.squashing = activations.get(squashing)
        self.beta = beta
        super().__init__(**kwargs)

    def call(self, y_true, dists):
        distances = tf.expand_dims(dists, axis=1)
        matcher = tf.equal(tf.expand_dims(y_true, axis=1),
                           self.prototype_labels)
        not_matcher = tf.logical_not(matcher)

        # Ragged-Tensors are required to support non-uniform prototype
        # distributions
        distances_to_wpluses = tf.ragged.boolean_mask(distances, matcher)
        distances_to_wminuses = tf.ragged.boolean_mask(distances, not_matcher)
        neg_distances_to_wpluses = tf.negative(distances_to_wpluses)
        neg_distances_to_wminuses = tf.negative(distances_to_wminuses)

        # Convert Ragged-Tensors back to normal Tensors:
        neg_distances_to_wpluses = neg_distances_to_wpluses.to_tensor(
            default_value=-tf.constant(float('inf')))
        neg_distances_to_wminuses = neg_distances_to_wminuses.to_tensor(
            default_value=-tf.constant(float('inf')))

        dpluses, _ = tf.nn.top_k(neg_distances_to_wpluses, k=1)
        dminuses, _ = tf.nn.top_k(neg_distances_to_wminuses, k=1)
        classifier = tf.math.divide(dpluses - dminuses, dpluses + dminuses)

        batch_loss = self.squashing(classifier, beta=self.beta)
        loss = batch_loss
        return loss

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
