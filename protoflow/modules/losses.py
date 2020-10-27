"""ProtoFlow Losses."""

import tensorflow as tf

from protoflow.functions import activations
from protoflow.functions.losses import get_dp_dm


class GLVQLossLabeless(tf.keras.losses.Loss):
    """GLVQ Loss which assumes that the last row of the inputs are the
    prototype labels.
    """
    def __init__(self,
                 squashing='beta_sigmoid',
                 beta=20,
                 minimize_dp=False,
                 **kwargs):
        self.squashing = activations.get(squashing)
        self.beta = beta
        self.minimize_dp = minimize_dp
        super().__init__(**kwargs)

    def get_dp_dm(self, y_true, output):
        distances = output[:-1]
        prototype_labels = output[-1]
        dp, dm = get_dp_dm(y_true, distances, prototype_labels)
        return dp, dm

    def call(self, y_true, output):
        dp, dm = self.get_dp_dm(y_true, output)

        # Regularization
        if self.minimize_dp:
            reg_term = dp
        else:
            reg_term = 0

        mu = (dp - dm) / (dp + dm)
        batch_loss = self.squashing(mu, beta=self.beta)
        return batch_loss + reg_term

    def get_config(self):
        base_config = super().get_config()
        config = {
            'squashing': activations.serialize(self.squashing),
            'beta': self.beta,
        }
        # Concatenate the dictionaries
        return {**base_config, **config}


class GLVQLoss(GLVQLossLabeless):
    """GLVQ Loss which stores the `prototype_labels`."""
    def __init__(self, prototype_labels, **kwargs):
        self.prototype_labels = prototype_labels
        super().__init__(**kwargs)

    def get_dp_dm(self, y_true, distances):
        dp, dm = get_dp_dm(y_true, distances, self.prototype_labels)
        return dp, dm

    def get_config(self):
        base_config = super().get_config()
        config = {
            'prototype_labels': self.prototype_labels,
        }
        # Concatenate the dictionaries
        return {**base_config, **config}
