"""ProtoFlow callbacks: called during model training."""

import tensorflow as tf

from protoflow.utils.utils import prettify_string, writelog


class LossHistory(tf.keras.callbacks.Callback):
    """Simple Callback that saves the loss after every epoch."""
    def __init__(self, **kwargs):
        self.losses = []
        """List of Losses"""
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs={}):
        """Save the loss after every epoch."""
        self.losses.append(logs.get('loss'))


class TerminateOnProtosNaN(tf.keras.callbacks.Callback):
    """Simple Callback that ends training on NaN."""
    def on_batch_end(self, epoch, logs={}):
        protos = self.model.distance.prototypes
        error_msg = """NaN in prototypes due to unstable learning dynamics.
        Try a different prototype initialization or decrease the number of
        prototypes. Just lowering the learning rate might sometimes be
        sufficient.
        """
        error_msg = prettify_string(error_msg)

        if True in tf.math.is_nan(protos).numpy():
            writelog(error_msg)
            writelog('Prototypes:', str(protos))
            raise ValueError(error_msg)
