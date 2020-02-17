"""ProtoFlow product layers."""

import tensorflow as tf

from protoflow.functions import products


class LpSIP(tf.keras.layers.Dense):
    """Semi-Inner Product layer based on the :math:`L_p` norm.

    Arguments:
        units: (int), number of units in the layer.

    Keyword arguments:
        p: (int), the :math:`p` in :math:`L_p`.
        soft: (int), computes the soft inner product if non-zero.
        trainable_p: (bool), sets parameter `p` as trainable if set to True.
    """
    def __init__(self, units, p=2, soft=0, trainable_p=False, **kwargs):
        super().__init__(units, **kwargs)
        self.p0 = p
        self.soft = soft
        self.trainable_p = trainable_p

    def build(self, input_shape):
        self.p = self.add_weight(name='p',
                                 shape=(1, ),
                                 initializer=tf.keras.initializers.Constant(
                                     self.p0),
                                 dtype=self.dtype,
                                 trainable=self.trainable_p)
        super().build(input_shape)

    def call(self, inputs):
        output = products.lp_sip(inputs,
                                 self.kernel,
                                 p=self.p,
                                 alpha=self.soft)
        if self.use_bias:
            # output += self.bias  # deprecated in favor of tf.nn.bias_add
            output = tf.nn.bias_add(output, self.bias, data_format='N...C')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({'trainable_p': self.trainable_p, 'soft': self.soft})
        return config
