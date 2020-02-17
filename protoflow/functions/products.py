"""ProtoFlow product functions."""

import tensorflow as tf


def soft_abs(x, alpha=10):
    """Soft absolute."""
    return 1 / alpha * tf.math.log(2 + tf.math.exp(-alpha * x) +
                                   tf.math.exp(alpha * x))


def lp_sip(x, w, p=2, alpha=0):
    """Compute the :math:`L_p` semi-inner-product between :math:`x` and :math:`w`.

    Expected dimension of x is 2.
    Expected dimension of w is 2.
    """
    if alpha:
        pnorm_x = tf.pow(
            tf.reduce_sum(tf.pow(soft_abs(x, alpha=alpha), p), axis=1),
            1.0 / p)
        x_raised_p_minus_1 = tf.pow(soft_abs(x, alpha), p - 1.0)
    else:
        # pnorm_x = tf.linalg.norm(x, ord=p, axis=1, keepdims=True)
        pnorm_x = tf.pow(tf.reduce_sum(tf.pow(x, p), axis=1, keepdims=True),
                         1.0 / p)
        x_raised_p_minus_1 = tf.pow(tf.abs(x), p - 1.0)
    sum_term = tf.matmul(x_raised_p_minus_1 * tf.sign(x), w)
    pdt = (1.0 / tf.pow(pnorm_x, p - 2.0)) * sum_term
    return pdt
