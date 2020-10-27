"""ProtoFlow constraints."""

import tensorflow as tf

from protoflow.functions import normalization


class Zero(tf.keras.constraints.Constraint):
    """Constrain the weights to be zero.

    Important:
        Only for testing. We do not expect this to be used in practice.
    """
    def __call__(self, w):
        return w * 0


class Positive(tf.keras.constraints.Constraint):
    """Constrain the weights to be positive."""
    def __call__(self, w):
        return w * tf.cast(tf.math.greater(w, 0.), tf.keras.backend.floatx())


class Negative(tf.keras.constraints.Constraint):
    """Constrain the weights to be negative."""
    def __call__(self, w):
        return w * tf.cast(tf.math.less(w, 0.), tf.keras.backend.floatx())


class TraceNormalization(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return normalization.trace_normalization(w)


class NullspaceNormalization(tf.keras.constraints.Constraint):
    def __init__(self, data, threshold=0.01):
        self.data = data
        self.threshold = threshold

    def __call__(self, w):
        return normalization.nullspace_normalization(
            w,
            self.data,
            self.threshold,
        )


# class Orthogonalization(tf.keras.constraints.Constraint):
#     def __call__(self, w):
#         return tf.keras.utils.norm_funcs.orthogonalization(w)
