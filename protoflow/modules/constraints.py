"""ProtoFlow constraints."""

import six
import tensorflow as tf
from tensorflow.keras.constraints import *  # noqa: F403, F401
from tensorflow.keras.utils import (deserialize_keras_object,
                                    serialize_keras_object)

from protoflow.functions import normalization


class TraceNormalization(tf.keras.constraints.Constraint):
    def __init__(self, epsilon=tf.keras.backend.epsilon()):
        self.epsilon = epsilon

    def __call__(self, w):
        return normalization.trace_normalization(w, epsilon=self.epsilon)

    def get_config(self):
        return {'epsilon': self.epsilon}


class OmegaNormalization(tf.keras.constraints.Constraint):
    """OmegaNormalization

    OmegaNormalization can be also applied after training for interpretation.

    Important:
        This constraint is not compatible with OmegaRegularizer. Do not apply
        both on the same time. Numerically instable!
    """
    def __init__(self, epsilon=tf.keras.backend.epsilon()):
        self.epsilon = epsilon

    def __call__(self, w):
        return normalization.omega_normalization(w, epsilon=self.epsilon)

    def get_config(self):
        return {'epsilon': self.epsilon}


class Orthogonalization(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.keras.utils.norm_funcs.orthogonalization(w)


class Positive(tf.keras.constraints.Constraint):
    """Constrains the weights to be positive."""
    def __call__(self, w):
        return w * tf.cast(tf.math.greater(w, 0.), tf.keras.backend.floatx())


class Negative(tf.keras.constraints.Constraint):
    """Constrains the weights to be negative."""
    def __call__(self, w):
        return w * tf.cast(tf.math.less(w, 0.), tf.keras.backend.floatx())


class Zero(tf.keras.constraints.Constraint):
    """Constrains the weights to be zero.

    Important:
        Only for testing. We do not expect this to be used in practice.
    """
    def __call__(self, w):
        return w * 0


# Aliases

trace_normalization = tracenorm = TraceNormalization
omega_normalization = omeganorm = OmegaNormalization
orthogonalization = Orthogonalization
positive = Positive
negative = Negative
zero = Zero


def serialize(constraint):
    return serialize_keras_object(constraint)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='ProtoFlow constraint')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret constraint identifier: ' +
                         str(identifier))
