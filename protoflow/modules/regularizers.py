"""ProtoFlow regularizers."""

import tensorflow as tf


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
        return -self.alpha * tf.reduce_sum(tf.math.log(tf.linalg.det(w)))

    def get_config(self):
        base_config = super().get_config()
        config = {"alpha": self.alpha}
        return {**base_config, **config}
