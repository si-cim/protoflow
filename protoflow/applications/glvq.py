"""Build and train GLVQ models with tf.keras."""

import tensorflow as tf

import protoflow as pf
from protoflow.applications.network import Network
from protoflow.functions.distances import squared_euclidean_distance


class _GLVQ(Network):
    """Abstract class that adds the GLVQ loss and implements the competition
    layer used by GLVQ-like models including GMLVQ and LVQMLN.

    """

    def compile(self,
                loss=None,
                squashing="sigmoid_beta",
                beta=10,
                minimize_dp=False,
                optimizer=tf.keras.optimizers.Adam(0.01),
                **kwargs):
        """Compile the model.

        Keyword Arguments:
            squashing (str) : Loss squashing function.
                (default: "sigmoid_beta")
            beta (float) : Beta value for the squashing function.
            optimizer : Optimizer to use for training.
                (default: tf.keras.optimizers.Adam(0.01))
        """
        plabels = self.prototype_layer.prototype_labels
        if loss is None:
            loss = pf.modules.losses.GLVQLoss(plabels,
                                              squashing=squashing,
                                              beta=beta,
                                              minimize_dp=minimize_dp)
        super().compile(loss=loss,
                        optimizer=optimizer,
                        metrics=[pf.metrics.wtac_accuracy(plabels)])

    def competition(self, distances):
        y_pred = pf.functions.wtac(distances,
                                   self.prototype_layer.prototype_labels)
        return y_pred


class GLVQ(_GLVQ):
    """Generalized Learning Vector Quantization (GLVQ) [sato1996]_.

    GLVQ is the basic network for most other LVQ models in ProtoFlow.

    Keyword Arguments:
        prototypes_per_class (int) : Number of prototypes in each class.
            (default: 1)
        prototype_distribution (list) : Use custom prototype
            distribution. (default: None)
        loss_squashing (str) : Loss squashing function.
            (default: "sigmoid_beta")
        prototype_initializer (str) : Method to use to set the initial
            prototype locations. (default: "mean")
    """

    def __init__(self,
                 nclasses,
                 input_dim,
                 prototypes_per_class=1,
                 prototype_initializer="zeros",
                 trainable_prototypes=True,
                 prototypes_dtype="float32",
                 prototype_constraint=None,
                 distance_fn=squared_euclidean_distance,
                 **kwargs):
        super().__init__(**kwargs)

        self.prototype_layer = pf.layers.Prototypes1D(
            nclasses=nclasses,
            prototypes_per_class=prototypes_per_class,
            prototype_initializer=prototype_initializer,
            trainable_prototypes=trainable_prototypes,
            prototype_constraint=prototype_constraint,
            dtype=prototypes_dtype,
        )
        self.distance_fn = distance_fn

        self.build(input_shape=(None, input_dim))

    def call(self, inputs):
        protos = self.prototype_layer(inputs)
        distances = self.distance_fn(inputs, protos)
        return distances
