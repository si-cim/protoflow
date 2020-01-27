"""ProtoFlow regularizers."""

import six
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import *  # noqa: F403, F401
from tensorflow.keras.utils import (deserialize_keras_object,
                                    serialize_keras_object)


class LogDeterminant(tf.keras.regularizers.Regularizer):
    """Computes :math:`ln(det(A))` or :math:`ln(det(A'*A))`.

    A is assumed as transposed matrix if it is a omega matrix. Thus we
    symmetrize over the subspace.

    .. TODO::
        Finish this docstring.
    """
    def __init__(self, alpha=0.005):
        self.alpha = alpha

    def __call__(self, w):
        with K.name_scope('log_determinant'):
            return -self.alpha * K.sum(K.log(tf.linalg.det(w)))

    def get_config(self):
        return {'alpha': self.alpha}


class OmegaRegularizer(tf.keras.regularizers.Regularizer):
    """Omega Regularization.

    .. TODO::
        Finish this docstring.
    """
    def __init__(self, alpha=0.005):
        self.alpha = alpha

    def __call__(self, w):
        with K.name_scope('omega_regularizer'):
            shape = K.int_shape(w)
            ndim = K.ndim(w)
            # find matrix minimal dimension
            if shape[-2] >= shape[-1]:
                axes = ndim - 2
            # this is needed to regularize if the points are projected
            # to a higher dimensional space
            else:
                axes = ndim - 1

            # batch matrices
            if ndim >= 3:
                w = K.batch_dot(w, w, [axes, axes])
            # non-batch
            else:
                if axes == 1:
                    w = K.dot(w, K.transpose(w))
                else:
                    w = K.dot(K.transpose(w), w)

            log_determinant = LogDeterminant(alpha=self.alpha)
            return log_determinant(w)

    def get_config(self):
        return {'alpha': self.alpha}


# Aliases (always calling the standard setting):


def log_determinant(w):
    regularizer = LogDeterminant()
    return regularizer(w)


def omega_regularizer(w):
    regularizer = OmegaRegularizer()
    return regularizer(w)


logdet = log_det = LogDeterminant
omegareg = omega_reg = OmegaRegularizer


def serialize(regularizer):
    return serialize_keras_object(regularizer)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='ProtoFlow regularizer')


def get(identifier):
    if identifier is None:
        return None
    elif isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        identifier = str(identifier)

        if identifier.islower():
            return deserialize(identifier)
        else:
            config = {'class_name': identifier, 'config': {}}
            return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret regularizer identifier: ' +
                         str(identifier))
