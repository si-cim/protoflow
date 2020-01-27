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
    accuracy = tf.reduce_mean(tf.cast(match, 'float32'))
    return accuracy * 100


def wtac_accuracy(prototype_labels):
    """Wrap a function that computes the accuracy for GLVQ-based models """
    def acc(y_true, dists):
        neg_distances = tf.negative(dists)
        _, winning_indices = tf.nn.top_k(neg_distances, k=1)
        winning_labels = tf.gather(prototype_labels, winning_indices)
        y_pred = winning_labels
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    return acc
