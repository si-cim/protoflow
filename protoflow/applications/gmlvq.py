"""Build and train GMLVQ models with tf.keras."""

import tensorflow as tf

import protoflow as pf
from protoflow.modules import constraints

from .glvq import _GLVQ


class GMLVQ(_GLVQ):
    """Generalized Matrix Learning Vector Quantization (GMLVQ) [schneider2009]_

    GMLVQ is a generalization of GRLVQ. Restricting the :math:`\\Omega` matrix
    to only contain non-zero values in the main diagonal yields GRLVQ.

    The distance measure in GMLVQ is taken as:

    .. math::
        d^\\Lambda (\\omega, \\xi) =
            (\\xi - \\omega)^T \\Lambda (\\xi - \\omega).

    By defining

    .. math::
        \\Lambda = \\Omega^T \\Omega

    it can be guaranteed that :math:`\\Lambda` is symmetric.
    """
    def __init__(self,
                 nclasses,
                 input_dim,
                 mapping_dim=None,
                 prototypes_per_class=1,
                 prototype_initializer="zeros",
                 matrix_initializer="glorot_uniform",
                 matrix_constraint=constraints.TraceNormalization(),
                 trainable_prototypes=True,
                 prototypes_dtype="float32",
                 distance_fn=pf.functions.distances.euclidean_distance,
                 **kwargs):
        super().__init__(**kwargs)

        if mapping_dim is None:
            mapping_dim = input_dim

        self.mapping_layer = tf.keras.layers.Dense(
            units=mapping_dim,
            activation=None,
            use_bias=False,
            kernel_initializer=matrix_initializer,
            kernel_constraint=matrix_constraint,
            input_dim=input_dim,
        )
        self.prototype_layer = pf.layers.Prototypes1D(
            nclasses=nclasses,
            prototypes_per_class=prototypes_per_class,
            prototype_initializer=prototype_initializer,
            trainable_prototypes=trainable_prototypes,
            dtype=prototypes_dtype,
        )
        self.distance_fn = distance_fn

        self.build(input_shape=(None, input_dim))

    @property
    def omega(self):
        """Returns the :math:`\\Omega` matrix.

        Returns:
            numpy.ndarray: Omega matrix
        """
        omega = self.mapping_layer.get_weights()[0]
        return omega

    def call(self, inputs):
        protos = self.prototype_layer(inputs)
        mapped_inputs = self.mapping_layer(inputs)
        mapped_protos = self.mapping_layer(protos)
        distances = self.distance_fn(mapped_inputs, mapped_protos)
        return distances
