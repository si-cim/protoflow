"""ProtoFlow loss functions."""

import tensorflow as tf


def get_match_and_rest(inputs, truth, labels, replacement='inf'):
    """Pick from each row of `inputs` the items that agree with the truth. The
    labels of the columns are given by the argument `labels`.

    The elements that do not agree are replaced by the argument `replacement`.

    """
    inf = tf.constant(float(replacement))
    truth = tf.cast(truth, inputs.dtype)
    match_mask = tf.equal(truth, labels)
    rest_mask = tf.logical_not(match_mask)
    match = tf.where(match_mask, inputs, inf)
    rest = tf.where(rest_mask, inputs, inf)
    return match, rest


def get_dp_dm(y_true, distances, prototype_labels):
    """Return the quantities :math:`d^{+}` and :math:`d^{-}`."""
    d_match, d_rest = get_match_and_rest(inputs=distances,
                                         truth=y_true,
                                         labels=prototype_labels,
                                         replacement='inf')
    dp = tf.keras.backend.min(d_match, axis=1, keepdims=True)
    dm = tf.keras.backend.min(d_rest, axis=1, keepdims=True)
    return dp, dm


def glvq_loss(y_true, distances, prototype_labels):
    """Evaluate the GLVQ classifier function for an entire batch."""
    dp, dm = get_dp_dm(y_true, distances, prototype_labels)
    mu = (dp - dm) / (dp + dm)
    return mu
