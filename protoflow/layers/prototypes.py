"""ProtoFlow Prototype layers."""

import itertools

import tensorflow as tf
from protoflow.modules import initializers


class Prototypes1D(tf.keras.layers.Layer):
    """Point Prototypes."""
    def __init__(self,
                 prototypes_per_class=1,
                 nclasses=None,
                 prototype_initializer='zeros',
                 trainable_prototypes=True,
                 dtype='float32',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'), )
        super().__init__(dtype=dtype, **kwargs)

        self.num_of_prototypes = prototypes_per_class * nclasses
        self.prototype_distribution = [prototypes_per_class] * nclasses
        self.prototype_initializer = initializers.get(prototype_initializer)
        self.trainable_prototypes = trainable_prototypes

    def build(self, input_shape):
        # Make a label list and flatten the list of lists using itertools
        llist = [[i] * n
                 for i, n in zip(range(len(self.prototype_distribution)),
                                 self.prototype_distribution)]
        flat_llist = list(itertools.chain(*llist))
        self.prototype_labels = tf.Variable(initial_value=flat_llist,
                                            dtype=self.dtype,
                                            trainable=False)
        self.prototypes = self.add_weight(
            name='prototypes',
            shape=(self.num_of_prototypes, input_shape[1]),
            dtype=self.dtype,
            initializer=self.prototype_initializer,
            trainable=self.trainable_prototypes)
        super().build(input_shape)

    def call(self, x):
        return self.prototypes

    def get_config(self):
        prototype_initializer = initializers.get(self.prototype_initializer)
        base_config = super().get_config()
        config = {
            'num_of_prototypes': self.num_of_prototypes,
            'prototype_labels': self.prototype_labels,
            'prototype_initializer': prototype_initializer,
            'trainable_prototypes': self.trainable_prototypes,
        }
        return {**base_config, **config}
