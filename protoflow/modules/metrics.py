"""ProtoFlow metrics."""

import tensorflow as tf


def accuracy_score(y_true, y_pred):
    """Accuracy Score. Compares predicted and true labels.

    Arguments:
        y_true (list(any)): True labels.
        y_pred (list(any)): Predicted labels.

    Returns:
        float: Accuracy in percentage.
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    match = y_true == y_pred
    accuracy = tf.reduce_mean(tf.cast(match, tf.float32))
    return accuracy * 100


def wtac_accuracy(prototype_labels=None):
    """Return a Winner-Takes-All-Competition (WTAC)-based accuracy metric
    to be used by Keras.

    If the prototype_labels are not passed, it is assumed that the last row of
    the output (distances) are the prototype_labels.

    """
    if prototype_labels is not None:

        def acc(y_true, distances):
            """Compute WTAC acc when labels are given."""
            winning_indices = tf.keras.backend.argmin(distances, axis=1)
            y_pred = tf.gather(prototype_labels, winning_indices)
            accuracy = accuracy_score(tf.reshape(y_true, shape=(-1, )), y_pred)
            return accuracy

        return acc

    else:

        def acc(y_true, output):
            """Compute WTAC acc when labels are NOT given, by
            assuming that the last row of `output` are the labels.
            """
            distances = output[:-1]
            prototype_labels = output[-1]
            winning_indices = tf.keras.backend.argmin(distances, axis=1)
            y_pred = tf.gather(prototype_labels, winning_indices)
            accuracy = accuracy_score(tf.reshape(y_true, shape=(-1, )), y_pred)
            return accuracy

        return acc
