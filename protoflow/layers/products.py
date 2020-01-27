"""ProtoFlow product layers."""

import tensorflow as tf

from protoflow.functions import products


class LpSIP(tf.keras.layers.Dense):
    def __init__(self, units, p=2, soft=0, **kwargs):
        super().__init__(units, **kwargs)
        self.p = p
        self.soft = soft

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
        config.update({'p': self.p, 'soft': self.soft})
        return config
