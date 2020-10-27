"""ProtoFlow normalization functions."""

import tensorflow as tf


def trace_normalization(mat, epsilon=1e-7):
    """Normalize the matrix such that the trace is close to 1.

    shape of `mat`: (input_dim, mapping_dim) or (k, input_dim, mapping_dim)
    """
    # mat = tf.cast(mat, dtype="float")
    matt = tf.transpose(mat)
    tr = tf.linalg.trace(matt @ mat, name="trace")
    normalized = tf.divide(mat, tf.math.sqrt(tr) + epsilon)
    return normalized


def nullspace_normalization(mat, data, threshold=0.01):
    """'Subtract' the nullspace from the matrix. Used on
    the :math:`\Omega` matrix in GMLVQ.

    shape of `mat`: (input_dim, mapping_dim)
    shape of `data`: (batch_size, input_dim)
    """
    C = tf.transpose(data) @ data
    E, V = tf.linalg.eig(C)
    E = tf.cast(E, dtype=mat.dtype)  # ignore the complex part
    V = tf.cast(V, dtype=mat.dtype)  # ignore the complex part
    V_sel = tf.transpose(V)[E < threshold]
    M = tf.transpose(V_sel) @ V_sel
    delta = tf.eye(mat.shape[1], dtype=mat.dtype) - M
    normalized = mat @ tf.transpose(delta)
    return normalized
