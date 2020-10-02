"""ProtoFlow Prototype layers."""

import itertools

import tensorflow as tf

from protoflow.modules import initializers


class _Prototypes(tf.keras.layers.Layer):
    """Base class for Prototype layers in ProtoFlow."""
    def __init__(self,
                 nclasses=None,
                 prototypes_per_class=1,
                 prototype_initializer='zeros',
                 trainable_prototypes=True,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = [kwargs.pop('input_dim')]
        super().__init__(**kwargs)

        self.num_of_prototypes = prototypes_per_class * nclasses
        self.prototype_distribution = [prototypes_per_class] * nclasses
        self.prototype_initializer = initializers.get(prototype_initializer)
        self.trainable_prototypes = trainable_prototypes
        self.prototypes = None
        self.prototype_labels = None

    def get_config(self):
        base_config = super().get_config()
        config = {
            'prototype_distribution': self.prototype_distribution,
            'trainable_prototypes': self.trainable_prototypes,
        }
        return {**base_config, **config}


class Prototypes1D(_Prototypes):
    """Point Prototypes."""
    def build(self, input_shape):
        # Make a label list and flatten the list of lists using itertools
        pdist = self.prototype_distribution
        label_list = [[i] * n for i, n in zip(range(len(pdist)), pdist)]
        plabels = list(itertools.chain(*label_list))
        self.prototype_labels = tf.Variable(initial_value=plabels,
                                            dtype=self.dtype,
                                            trainable=False)
        self.prototypes = self.add_weight(
            name='prototypes',
            shape=(self.num_of_prototypes, input_shape[1]),
            dtype=self.dtype,
            initializer=self.prototype_initializer,
            trainable=self.trainable_prototypes)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.prototypes
