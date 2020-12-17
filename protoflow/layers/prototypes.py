"""ProtoFlow Prototype layers."""

import itertools

import tensorflow as tf

from protoflow.modules import initializers

from tensorflow.python.keras import constraints


class _Prototypes(tf.keras.layers.Layer):
    """Base class for Prototype layers in ProtoFlow."""

    def __init__(self,
                 nclasses=None,
                 prototypes_per_class=1,
                 prototype_distribution=None,
                 prototype_initializer="zeros",
                 prototype_constraint=None,
                 trainable_prototypes=True,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = [kwargs.pop('input_dim')]
        super().__init__(**kwargs)

        self.nclasses = nclasses
        self.prototypes_per_class = prototypes_per_class
        self.prototype_distribution = [prototypes_per_class] * nclasses
        if prototype_distribution:
            # override if `prototype_distribution` is available
            assert self.nclasses == len(prototype_distribution)
            self.prototype_distribution = prototype_distribution
        self.prototype_initializer = initializers.get(prototype_initializer)
        self.prototype_constraint = constraints.get(prototype_constraint)
        self.trainable_prototypes = trainable_prototypes

        # Make a label list and flatten the list of lists using itertools
        pdist = self.prototype_distribution
        label_list = [[i] * n for i, n in zip(range(len(pdist)), pdist)]
        plabels = list(itertools.chain(*label_list))

        self.prototypes = None
        # self.prototype_labels = tf.Variable(initial_value=plabels,
        #                                     dtype=self.dtype,
        #                                     trainable=False)
        self.prototype_labels = tf.constant(value=plabels, dtype=self.dtype)

    def get_config(self):
        """Save everything you need to rebuild an identical Python object."""
        base_config = super().get_config()
        config = {
            'nclasses': self.nclasses,
            'prototypes_per_class': self.prototypes_per_class,
            'prototype_distribution': self.prototype_distribution,
            'trainable_prototypes': self.trainable_prototypes,
        }
        return {**base_config, **config}


class Prototypes1D(_Prototypes):
    """Point Prototypes."""

    def build(self, input_shape):
        num_of_prototypes = sum(self.prototype_distribution)
        self.prototypes = self.add_weight(
            name='prototypes',
            shape=(num_of_prototypes, input_shape[-1]),
            dtype=self.dtype,
            initializer=self.prototype_initializer,
            constraint=self.prototype_constraint,
            trainable=self.trainable_prototypes)
        super().build(input_shape)

    def call(self, inputs):
        return self.prototypes


class AppendPrototypes1D(Prototypes1D):
    """Append the prototypes to the inputs and pass to the next layer.

    A hack that makes it possible to use a `tf.keras.Sequential` model
    to implement LVQ-like architectures.

    `shape_transformation`: Callable that is to be applied to get a matrix.
    """

    def __init__(self, shape_transformation=None, **kwargs):
        self.shape_transformation = shape_transformation or tf.identity
        super().__init__(**kwargs)

    def call(self, inputs):
        """Append trainable prototypes to the inputs and pass on.

        `return (inputs, self.prototypes)` is not allowed by `tf.keras.Sequential`.
        All layers in a Sequential model should have a single output tensor.
        For multi-output layers, use the functional API.
        """
        inputs = self.shape_transformation(inputs)
        return tf.concat([inputs, self.prototypes],
                         axis=0,
                         name="append_prototypes")
