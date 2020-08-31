"""ProtoFlow distance layers."""

import tensorflow as tf

from protoflow.functions import distances as dists
from protoflow.modules import constraints, initializers, regularizers


class Distance(tf.keras.layers.Layer):
    """Abstract distance layer to be subclassed.

    Arguments:
        num_of_prototypes: Positive integer, number of prototypes.
        prototype_labels: (list), class labels the prototype vectors.

    Keyword arguments:
        prototype_initializer: Initializer for the `prototype` weights matrix
            (see [initializers](../modules/initializers.md)).
        prototype_regularizer: Regularizer function applied to
            the `prototype` weights matrix
            (see [regularizer](../modules/regularizers.md)).
        prototype_constraint: Constraint function applied to
            the `prototype` weights matrix
            (see [constraints](../modules/constraints.md)).
    """
    def __init__(self,
                 num_of_prototypes,
                 prototype_labels,
                 prototype_initializer='zeros',
                 prototype_regularizer=None,
                 prototype_constraint=None,
                 trainable_prototypes=True,
                 dtype='float32',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'), )

        super().__init__(dtype=dtype, **kwargs)

        self.num_of_prototypes = num_of_prototypes
        self.prototype_labels = prototype_labels
        self.prototype_initializer = initializers.get(prototype_initializer)
        self.prototype_regularizer = regularizers.get(prototype_regularizer)
        self.prototype_constraint = constraints.get(prototype_constraint)
        self.trainable_prototypes = trainable_prototypes

    def build(self, input_shape):
        prototype_dim = input_shape[1]
        self.prototypes = self.add_weight(
            name='prototypes',
            shape=(self.num_of_prototypes, prototype_dim),
            dtype=self.dtype,
            initializer=self.prototype_initializer,
            regularizer=self.prototype_regularizer,
            constraint=self.prototype_constraint,
            trainable=self.trainable_prototypes)
        super().build(input_shape)

    def call(self, x):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_of_prototypes)

    def get_config(self):
        prototype_initializer = initializers.get(self.prototype_initializer)
        prototype_regularizer = regularizers.get(self.prototype_regularizer)
        prototype_constraint = constraints.get(self.prototype_constraint)
        base_config = super().get_config()
        config = {
            'num_of_prototypes': self.num_of_prototypes,
            'prototype_labels': self.prototype_labels,
            'prototype_initializer': prototype_initializer,
            'prototype_regularizer': prototype_regularizer,
            'prototype_constraint': prototype_constraint,
            'trainable_prototypes': self.trainable_prototypes,
        }
        return {**base_config, **config}


class ManhattanDistance(Distance):
    """Compute the Manhattan distance between an incoming
    batch of data :math:`x` and the prototypes :math:`w`
    of the layer.

    Returns:
        tensorflow.Tensor: distances
    """
    def call(self, x):
        d = dists.lpnorm_distance(x, self.prototypes, p=1)
        return d


class EuclideanDistance(Distance):
    """Compute the Euclidean distance between an incoming
    batch of data :math:`x` and the prototypes :math:`w`
    of the layer.

    Returns:
        tensorflow.Tensor: distances
    """
    def call(self, x):
        d = dists.euclidean_distance(x, self.prototypes)
        return d


class LpNormDistance(Distance):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p

    def call(self, x):
        d = dists.lpnorm_distance(x, self.prototypes, self.p)
        return d

    def get_config(self):
        base_config = super().get_config()
        config = {
            'p': self.p,
        }
        return {**base_config, **config}


class MatrixEuclideanDistance(Distance):
    """Compute the scaled Euclidean distance between an incoming
    batch of data :math:`x` and the prototypes :math:`w` of the
    layer.

    Supports localized matrices via 'local' `matrix_scope`.

    Keyword arguments:
        mapping_dim: (int) Optionally learn a limited rank matrix.
            Default None.

    Returns:
        tensorflow.Tensor: distances
    """
    def __init__(self,
                 num_of_prototypes,
                 prototype_labels,
                 mapping_dim,
                 trainable_matrix=True,
                 matrix_scope='global',
                 matrix_initializer='rand',
                 matrix_regularizer=None,
                 matrix_constraint=None,
                 **kwargs):
        super().__init__(num_of_prototypes, prototype_labels, **kwargs)
        self.mapping_dim = mapping_dim
        self.trainable_matrix = trainable_matrix
        self.matrix_scope = matrix_scope
        self.matrix_initializer = initializers.get(matrix_initializer)
        self.matrix_regularizer = regularizers.get(matrix_regularizer)
        self.matrix_constraint = constraints.get(matrix_constraint)

    def build(self, input_shape):
        if not self.mapping_dim:
            self.mapping_dim = input_shape[1]
        if self.matrix_scope == 'global':
            omega_shape = (input_shape[1], self.mapping_dim)
        if self.matrix_scope == 'local':
            omega_shape = (self.num_of_prototypes, input_shape[1],
                           self.mapping_dim)

        self.omega = self.add_weight(name=f'{self.matrix_scope}_omega',
                                     shape=omega_shape,
                                     dtype=self.dtype,
                                     initializer=self.matrix_initializer,
                                     regularizer=self.matrix_regularizer,
                                     constraint=self.matrix_constraint,
                                     trainable=self.trainable_matrix)
        super().build(input_shape)

    def call(self, x):
        if self.matrix_scope == 'global':
            d = dists.omega_distance(x, self.prototypes, self.omega)
        if self.matrix_scope == 'local':
            d = dists.lomega_distance(x, self.prototypes, self.omega)
        return d

    def get_config(self):
        matrix_initializer = initializers.serialize(self.matrix_initializer)
        matrix_regularizer = regularizers.serialize(self.matrix_regularizer)
        matrix_constraint = constraints.serialize(self.matrix_constraint)
        base_config = super().get_config()
        config = {
            'mapping_dim': self.mapping_dim,
            'trainable_matrix': self.trainable_matrix,
            'matrix_scope ': self.matrix_scope,
            'matrix_initializer': matrix_initializer,
            'matrix_regularizer': matrix_regularizer,
            'matrix_constraint': matrix_constraint,
        }
        return {**base_config, **config}


class SquaredEuclideanDistance(EuclideanDistance):
    """Compute the squared Euclidean distance between an incoming
    batch of data :math:`x` and the prototypes :math:`w` of the
    layer.

    Returns:
        tensorflow.Tensor: distances
    """
    def call(self, x):
        d = dists.squared_euclidean_distance(x, self.prototypes)
        return d


# Aliases
OmegaDistance = MatrixDistance = MatrixEuclideanDistance
SED = SquaredEuclideanDistance
