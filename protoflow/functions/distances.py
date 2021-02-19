"""ProtoFlow distance functions."""

import tensorflow as tf


def squared_euclidean_distance(x, y):
    """Compute the squared Euclidean distance between :math:`x` and :math:`y`.

    Expected dimension of x is 2 (batch_dim, input_dim).
    Expected dimension of y is 2 (nprotos, input_dim).
    """
    expanded_x = tf.expand_dims(x, axis=1)
    batchwise_difference = tf.subtract(y, expanded_x)
    differences_raised = tf.math.pow(batchwise_difference, 2)
    distances = tf.reduce_sum(input_tensor=differences_raised, axis=2)
    return distances


def euclidean_distance(x, y):
    """Compute the Euclidean distance between :math:`x` and :math:`y`.

    Expected dimension of x is 2 (batch_dim, input_dim).
    Expected dimension of y is 2 (nprotos, input_dim).
    """
    distances_raised = squared_euclidean_distance(x, y)
    distances = tf.math.pow(distances_raised, 1.0 / 2.0)
    return distances


def lpnorm_distance(x, y, p):
    """Compute :math:`{\\langle x, y \\rangle}_p`.

    Expected dimension of x is 2 (batch_dim, input_dim).
    Expected dimension of y is 2 (nprotos, input_dim).
    """
    expanded_x = tf.expand_dims(x, axis=1)
    batchwise_difference = tf.subtract(y, expanded_x)
    differences_raised = tf.math.pow(batchwise_difference, p)
    distances_raised = tf.reduce_sum(input_tensor=differences_raised, axis=2)
    distances = tf.math.pow(distances_raised, 1.0 / p)
    return distances


def omega_distance(x, y, omega):
    """Omega distance.

    Compute :math:`{\\langle \\Omega x, \\Omega y \\rangle}_p`

    Expected dimension of x is 2 (batch_dim, input_dim).
    Expected dimension of y is 2 (nprotos, input_dim).
    Expected dimension of omega is 2 (input_dim, mapping_dim).
    """
    projected_x = x @ omega
    projected_y = y @ omega
    distances = squared_euclidean_distance(projected_x, projected_y)
    return distances


def lomega_distance(x, y, omegas):
    """Localized Omega distance.

    Compute :math:`{\\langle \\Omega_k x, \\Omega_k y_k \\rangle}_p`

    Expected dimension of x is 2 (batch_dim, input_dim).
    Expected dimension of y is 2 (nprotos, input_dim).
    Expected dimension of omegas is 3 (nprotos, input_dim, mapping_dim).
    """
    projected_x = x @ omegas
    projected_y = tf.keras.backend.batch_dot(y, omegas)
    expanded_y = tf.expand_dims(projected_y, axis=1)
    batchwise_difference = tf.subtract(expanded_y, projected_x)
    differences_squared = tf.math.pow(batchwise_difference, 2)
    distances = tf.reduce_sum(input_tensor=differences_squared, axis=2)
    distances = tf.keras.backend.permute_dimensions(distances, (1, 0))
    return distances
