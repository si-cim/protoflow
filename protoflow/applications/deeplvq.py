"""Module to build, train and evaluate Deep LVQ models with Keras."""

import numpy as np
import tensorflow as tf

from protoflow import layers, modules
from protoflow.applications.glvq import GLVQ
from protoflow.functions import activations
from protoflow.modules import initializers


class DeepLVQ(GLVQ):
    """Deep Learning Vector Quantization (Deep-LVQ) [SHRV18]_.

    Deep-LVQ fuses Multi-Layer Perceptron (MLP) Networks with LVQ.

    Arguments:
        hidden_units (list(int)): List of sizes for hidden layers.

    Keyword Arguments:
        prototypes_per_class (int): Number of prototypes in each class.
        layer_activations (list): Activation functions of the hidden layers.
        kernel_initializers (list): Initializers to use for the hidden layers.
    """
    def __init__(self,
                 hidden_units,
                 prototypes_per_class=1,
                 layer_activations=['sigmoid'],
                 layer_biases=[True],
                 kernel_initializers=['glorot_normal'],
                 **kwargs):
        super().__init__(prototypes_per_class, **kwargs)
        self.hidden_units = hidden_units
        if len(layer_activations) == 1:
            layer_activations = layer_activations * len(hidden_units)
        if len(layer_biases) == 1:
            layer_biases = layer_biases * len(hidden_units)
        if len(kernel_initializers) == 1:
            kernel_initializers = kernel_initializers * len(hidden_units)
        if len(hidden_units) != len(layer_activations):
            raise ValueError(f'You provided {len(hidden_units)} '
                             f'hidden_units, but '
                             f'{len(layer_activations)} layer_activations.')
        if len(hidden_units) != len(kernel_initializers):
            raise ValueError(f'You provided {len(hidden_units)} '
                             f'hidden_layer_sizes, but '
                             f'{len(layer_activations)} layer_activations.')
        self.layer_activations = [
            activations.get(a) for a in layer_activations
        ]
        self.layer_biases = layer_biases
        self.kernel_initializers = [
            initializers.get(ki) for ki in kernel_initializers
        ]

    def build(self,
              x_train,
              y_train,
              custom_prototypes=None,
              prototype_initializer='mean'):
        """Initialize prototypes and build the Layers.

        Arguments:
            x_train : Training inputs.
            y_train : Training targets.

        Keyword Arguments:
            custom_prototypes (array-like) : Distance layer is reinitialized
                with this. Ignored if None. (default: None)
            prototype_initializer (str) : Method to use to set the initial
                prototype locations. (default: 'mean')
        """
        self._check_validity(x_train, y_train)

        classlabels = np.unique(y_train)
        self.num_classes = classlabels.size
        if not self.prototype_distribution:
            self.prototype_distribution = [self.prototypes_per_class
                                           ] * self.num_classes
        else:
            self._check_prototype_distribution()

        input_dims = [x_train.shape[1]] + self.hidden_units
        # Define projection.
        projection_layers = []
        for i, units in enumerate(self.hidden_units):
            projection_layers.append(
                tf.keras.layers.Dense(
                    units,
                    use_bias=self.layer_biases[i],
                    kernel_initializer=self.kernel_initializers[i],
                    activation=self.layer_activations[i],
                    input_dim=input_dims[0],
                    trainable=True,
                    dtype=self.dtype,
                    name=f'projection_{i}',
                ))
        self.projection = tf.keras.models.Sequential(projection_layers,
                                                     name='proj')
        x_proj = self.projection(x_train).numpy()

        # Compute initial prototype locations.
        if prototype_initializer in modules.PROTOTYPE_INITIALIZERS:
            init_getter = modules.PROTOTYPE_INITIALIZERS[prototype_initializer]
            prototype_initializer = init_getter(x_proj, y_train,
                                                self.prototype_distribution,
                                                self.verbose)
        else:
            prototype_initializer = modules.initializers.get(
                prototype_initializer)
        prototype_labels = np.empty(shape=(0, ), dtype=y_train.dtype)
        for label, num in zip(classlabels, self.prototype_distribution):
            prototype_labels = np.append(prototype_labels, [label] * num)

        # Define other layers.
        self.distance = layers.SquaredEuclideanDistance(
            num_of_prototypes=np.sum(self.prototype_distribution),
            prototype_dim=x_proj.shape[1],
            prototype_labels=prototype_labels,
            prototype_initializer=prototype_initializer,
            dtype=self.dtype,
            name='sqeuclid',
        )
        if custom_prototypes is not None:
            weights = self.distance.get_weights()
            weights[-1] = custom_prototypes
            self.distance.set_weights(weights)
        self.competition = layers.WTAC(
            prototype_labels=prototype_labels,
            dtype=self.dtype,
            name='wtac',
        )

        # Build model
        self.model = tf.keras.models.Sequential(
            [self.projection, self.distance])
        # Hook the layers to the model under names used by the callbacks
        self.model.projection = self.projection
        self.model.competition = self.competition
        self.model.distance = self.distance

        self.built = True
