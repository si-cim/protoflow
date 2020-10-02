"""ProtoFlow distance functions."""

import tensorflow as tf


def wtac(distances, labels):
    """Winner-Takes-All Competition."""
    winning_indices = tf.keras.backend.argmin(distances, axis=1)
    y_pred = tf.gather(labels, winning_indices)
    return y_pred


def knnc(distances, one_hot_labels, k=1):
    """k-Nearest-Neighbors Competition."""
    _, winning_indices = tf.nn.top_k(-distances, k=k)
    top_k_labels = tf.gather(one_hot_labels, winning_indices)
    predictions_sum = tf.reduce_sum(input_tensor=top_k_labels, axis=1)
    y_pred = tf.argmax(input=predictions_sum, axis=1)
    return y_pred
