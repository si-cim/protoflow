"""Module to build, train and evaluate GMLVQ models with Keras."""

import numpy as np
import tensorflow as tf

from protoflow import layers, modules
from protoflow.applications.glvq import GLVQ


class GMLVQ(GLVQ):
    """Generalized Matrix Learning Vector Quantization (GMLVQ) [ScBH09]_

    GMLVQ is an extention of GRLVQ.

    By introducing a Matrix into the distance measure with

    .. math::
        d^\\Lambda (\\omega, \\xi) =
            (\\xi - \\omega)^T \\Lambda (\\xi - \\omega).

    By defining

    .. math::
        \\Lambda = \\Omega^T \\Omega

    it can be guaranteed that :math:`\\Lambda` is symmetric.

    During learning, GMLVQ applies updates the prototypes and the
    :math:`\\Omega` matrix. The matrix :math:`\\Lambda` can be interpreted
    as relevance matrix.

    .. TODO::
        Check this explanation.
    """
    def __init__(self, prototypes_per_class=1, **kwargs):
        super().__init__(prototypes_per_class, **kwargs)

    @property
    def omega(self):
        """Returns the :math:`\\Omega` matrix.

        Returns:
            numpy.ndarray: Omega matrix
        """
        omega = self.distance.omega.numpy()
        return omega

    def build(self,
              x_train,
              y_train,
              custom_prototypes=None,
              prototype_initializer='mean',
              matrix_initializer='eye'):
        """Initialize prototypes and build the Layers.

        Arguments:
            x_train : Training inputs.
            y_train : Training targets.

        Keyword Arguments:
            custom_prototypes (array-like) : Distance layer is reinitialized
                with this. Ignored if None. (default: None)
            prototype_initializer (str) : Method to use to set the initial
                prototype locations. (default: 'mean')
            matrix_initializer (str) : Method to use to set the initial
                :math:`\\Omega` matrix. (default: 'eye')
        """
        self._check_validity(x_train, y_train)

        # Compute initial prototype locations.
        classlabels = np.unique(y_train)
        self.num_classes = classlabels.size
        if not self.prototype_distribution:
            self.prototype_distribution = [self.prototypes_per_class
                                           ] * self.num_classes
        else:
            self._check_prototype_distribution()

        if prototype_initializer in modules.PROTOTYPE_INITIALIZERS:
            init_getter = modules.PROTOTYPE_INITIALIZERS[prototype_initializer]
            prototype_initializer = init_getter(x_train, y_train,
                                                self.prototype_distribution,
                                                self.verbose)
        else:
            prototype_initializer = modules.initializers.get(
                prototype_initializer)
        prototype_labels = np.empty(shape=(0, ), dtype=y_train.dtype)
        for label, num in zip(classlabels, self.prototype_distribution):
            prototype_labels = np.append(prototype_labels, [label] * num)

        # Define layers.
        self.projection = tf.keras.layers.Dense(
            x_train.shape[1],
            use_bias=False,
            input_shape=(x_train.shape[1], ),
            dtype=self.dtype,
            kernel_initializer='identity',
            trainable=False,
            name='proj',
        )
        matrix_initializer = modules.initializers.get(matrix_initializer)
        self.distance = layers.OmegaDistance(
            num_of_prototypes=np.sum(self.prototype_distribution),
            input_dim=x_train.shape[1],
            mapping_dim=x_train.shape[1],
            prototype_labels=prototype_labels,
            prototype_initializer=prototype_initializer,
            matrix_initializer=matrix_initializer,
            matrix_regularizer=modules.OmegaRegularizer(alpha=0.005),
            dtype=self.dtype,
            name='omega',
        )
        self.competition = layers.WTAC(
            prototype_labels=prototype_labels,
            dtype=self.dtype,
            name='wtac',
        )

        # Build model
        self.model = tf.keras.models.Sequential(
            [self.projection, self.distance])

        if custom_prototypes is not None:
            self.set_prototypes(custom_prototypes)

        # Hook the layers to the model under names used by the callbacks
        self.model.projection = self.projection
        self.model.competition = self.competition
        self.model.distance = self.distance

        self.built = True
