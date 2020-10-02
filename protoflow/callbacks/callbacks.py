"""ProtoFlow callbacks: called during model training."""

import warnings

import tensorflow as tf


class TerminateOnProtosNaN(tf.keras.callbacks.Callback):
    """Simple Callback that ends training on NaN."""
    def on_batch_end(self, epoch, logs={}):
        protos = self.model.distance.prototypes
        error_msg = "NaN in prototypes due to unstable learning dynamics. "\
                    "Try a different prototype initialization or decrease "\
                    "the number of prototypes. Or even simply lowering "\
                    "the learning rate might sometimes be sufficient."
        if True in tf.math.is_nan(protos).numpy():
            warnings.warn(error_msg)
            raise ValueError(error_msg)
