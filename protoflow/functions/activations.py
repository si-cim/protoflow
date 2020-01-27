"""ProtoFlow activation functions."""

import warnings

import six
import tensorflow as tf
from tensorflow.keras.activations import *  # noqa: F403, F401
from tensorflow.keras.layers import Layer
import tensorflow.keras.activations
from tensorflow.keras.utils import deserialize_keras_object


def identity(input):
    """:math:`f(x) = x`"""
    return input


def sigmoid_beta(input, beta=10):
    """:math:`f(x) = \\frac{1}{1 + e^{-\\beta x}}`

    Keyword Arguments:
        beta (float): Parameter :math:`\\beta`
    """
    out = 1 / (1 + tf.exp(-beta * input))
    return out


def swish_beta(input, beta=0.1):
    """:math:`f(x) = \\frac{x}{1 + e^{-\\beta x}}`

    Keyword Arguments:
        beta (float): Parameter :math:`\\beta`
    """
    out = input * sigmoid_beta(input, beta=beta)
    return out


# Aliases

linear = identity
sigmoidb = beta_sigmoid = sigmoid_beta
swish = swishb = beta_siwsh = swish_beta


def serialize(activation):
    return activation.__name__


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='activation functions')


def get(identifier):
    """Get the `identifier` activation function.

    # Arguments
        identifier: None or str, name of the function.

    # Returns
        The activation function, `linear` if `identifier` is None.

    # Raises
        ValueError if unknown identifier.
    """
    if identifier is None:
        return linear
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        if isinstance(identifier, Layer):
            warnings.warn(
                'Do not pass a layer instance (such as {identifier}) as the '
                'activation argument of another layer. Instead, advanced '
                'activation layers should be used just like any other '
                'layer in a model.'.format(
                    identifier=identifier.__class__.__name__))
        return identifier
    else:
        raise ValueError(
            'Could not interpret '
            'activation function identifier:', identifier)
