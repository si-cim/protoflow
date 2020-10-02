"""Module to build, train and evaluate Deep LVQ models with Keras."""

import tensorflow as tf

import protoflow as pf
from protoflow.functions import activations
from protoflow.modules import initializers

from .glvq import _GLVQ


class DeepLVQ(_GLVQ):
    """Deep Learning Vector Quantization (Deep-LVQ) [saralajew2018]_.

    Deep-LVQ fuses Multi-Layer Perceptron (MLP) Networks with LVQ.

    Arguments:
        hidden_units (list(int)): List of sizes for hidden layers.

    Keyword Arguments:
        prototypes_per_class (int): Number of prototypes in each class.
        layer_activations (list): Activation functions of the hidden layers.
        kernel_initializers (list): Initializers to use for the hidden layers.
    """
    def __init__(self,
                 nclasses,
                 input_dim,
                 hidden_units,
                 prototypes_per_class=1,
                 prototype_initializer="zeros",
                 layer_activations=["sigmoid"],
                 layer_biases=[True],
                 kernel_initializers=["glorot_normal"],
                 trainable_prototypes=True,
                 prototypes_dtype="float32",
                 distance_fn=pf.functions.distances.euclidean_distance,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        if len(layer_activations) == 1:
            layer_activations = layer_activations * len(hidden_units)
        if len(layer_biases) == 1:
            layer_biases = layer_biases * len(hidden_units)
        if len(kernel_initializers) == 1:
            kernel_initializers = kernel_initializers * len(hidden_units)
        if len(hidden_units) != len(layer_activations):
            raise ValueError(f"You provided {len(hidden_units)} "
                             f"hidden_units, but "
                             f"{len(layer_activations)} layer_activations.")
        if len(hidden_units) != len(kernel_initializers):
            raise ValueError(f"You provided {len(hidden_units)} ",
                             f"hidden_layer_sizes, but ",
                             f"{len(layer_activations)} layer_activations.")
        self.layer_biases = layer_biases
        self.layer_activations = [
            activations.get(a) for a in layer_activations
        ]
        self.kernel_initializers = [
            initializers.get(ki) for ki in kernel_initializers
        ]

        self.mapping_layers = []
        for i, units in enumerate(self.hidden_units):
            self.mapping_layers.append(
                tf.keras.layers.Dense(
                    units,
                    use_bias=self.layer_biases[i],
                    kernel_initializer=self.kernel_initializers[i],
                    activation=self.layer_activations[i],
                    trainable=True,
                    dtype=self.dtype,
                    name=f"mapping_{i}",
                ))

        self.prototype_layer = pf.layers.Prototypes1D(
            nclasses=nclasses,
            prototypes_per_class=prototypes_per_class,
            prototype_initializer=prototype_initializer,
            trainable_prototypes=trainable_prototypes,
            dtype=prototypes_dtype,
        )
        self.distance_fn = distance_fn

        self.build(input_shape=(None, input_dim))

    def call(self, inputs):
        mapped_inputs = inputs + 0
        for mapping_layer in self.mapping_layers:
            mapped_inputs = mapping_layer(mapped_inputs)
        protos = self.prototype_layer(mapped_inputs)
        distances = self.distance_fn(mapped_inputs, protos)
        return distances
