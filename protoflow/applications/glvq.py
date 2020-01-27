"""Module to build, train and evaluate GLVQ models with Keras."""

import numpy as np
import tensorflow as tf

from protoflow import layers, modules, utils
from protoflow.applications.network import Network
from protoflow.callbacks import callbacks as pfcallbacks


class GLVQ(Network):
    """Generalized Learning Vector Quantization (GLVQ) [SaYa96]_.

    GLVQ is the basic network for most other LVQ models in ProtoFlow.

    Keyword Arguments:
        prototypes_per_class (int) : Number of prototypes in each class.
            (default: 1)
        prototype_distribution (list) : Use custom prototype
            distribution. (default: None)
        loss_squashing (str) : Loss squashing function.
            (default: 'sigmoid_beta')
    """
    def __init__(self,
                 prototypes_per_class=1,
                 prototype_distribution=None,
                 loss_squashing='sigmoid_beta',
                 **kwargs):
        super().__init__(**kwargs)
        self.prototypes_per_class = prototypes_per_class
        self.prototype_distribution = prototype_distribution
        self.loss_squashing = loss_squashing

        if prototype_distribution:
            error_msg = 'Invalid prototype distribution: Must only contain '\
                'positive values.'
            for i in prototype_distribution:
                if i <= 0:
                    raise ValueError(error_msg)
            self.prototype_per_class = None
        self.prototype_distribution = prototype_distribution

    @property
    def prototypes(self):
        """Get the prototypes from the distance layer.

        Returns:
            numpy.ndarray: prototypes
        """
        prototypes = self.distance.prototypes.numpy()
        return prototypes

    def set_prototypes(self, custom_prototypes):
        """Set the weights in the distance layer."""
        weights = self.distance.get_weights()
        weights[-1] = custom_prototypes
        self.distance.set_weights(weights)

    @property
    def prototype_labels(self):
        """Get the prototype labels form the distance layer.

        Returns:
            list: Class labels of prototypes.
        """
        prototype_labels = self.distance.prototype_labels
        return prototype_labels

    @property
    def num_of_prototypes(self):
        """Get the total number of prototypes from
        the distance layer.

        Returns:
            int: Total number of prototypes.
        """
        num_of_prototypes = len(self.prototype_labels)
        return num_of_prototypes

    def _check_prototype_distribution(self):
        """Assert correctness of the prototype distribution."""
        if len(self.prototype_distribution) != self.num_classes:
            error_msg = f"""Provided prototype distribution
            {self.prototype_distribution} does not match the number
            of classes ({self.num_classes}) in the training data.
            """
            raise ValueError(utils.prettify_string(error_msg))

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

        self.projection = tf.keras.layers.Dense(
            x_train.shape[1],
            use_bias=False,
            input_shape=(x_train.shape[1], ),
            dtype=self.dtype,
            kernel_initializer='identity',
            trainable=False,
            name='proj',
        )
        self.distance = layers.SquaredEuclideanDistance(
            num_of_prototypes=np.sum(self.prototype_distribution),
            prototype_dim=x_train.shape[1],
            prototype_labels=prototype_labels,
            prototype_initializer=prototype_initializer,
            dtype=self.dtype,
            name='sqeuclid',
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

    def _compile(self, learning_rate=0.01, run_eagerly=False, **kwargs):
        """Initialize loss function (GLVQLoss) and optimizer (Adam)
        and compile the built model.

        Keyword Arguments:
            learning_rate (float): Learning rate of the Adam
                optimizer. (default: 0.01)
            run_eagerly (bool): Use eager execution. (default: False)
        """
        learning_rate = kwargs.pop('lr', learning_rate)
        opt = tf.keras.optimizers.Adam(lr=learning_rate)
        loss = modules.GLVQLoss(prototype_labels=self.prototype_labels,
                                squashing=self.loss_squashing)
        metrics = [modules.wtac_accuracy(self.prototype_labels)]
        self.model.compile(
            optimizer=opt,
            loss=loss,
            run_eagerly=run_eagerly,
            metrics=metrics,
        )

    def predict(self, inputs):
        """Predict the class labels via Winner-Takes-All Competition (WTAC).

        Returns:
            Predicted class labels.
        """
        d = self.model(inputs)
        y_pred = self.competition(d)
        return y_pred.numpy()

    def fit(self,
            x_train,
            y_train,
            batch_size=32,
            learning_rate=0.01,
            shuffle=True,
            callbacks=[],
            run_eagerly=False,
            **kwargs):
        """Initialize callbacks and start Keras fit for Training.

        Arguments:
            x_train : Training inputs.
            y_train : Training targets.

        Keyword Arguments:
            batch_size (int): Size of training batches. (default: 32)
            learning_rate (float): Learning rate (same as lr). (default: 0.01)
            shuffle (bool): Shuffle the training set. (default: True)
            callbacks (list): List of callbacks. (default: [])
            run_eagerly (bool): Use eager execution. (default: False)
        """
        learning_rate = kwargs.pop('lr', learning_rate)

        if not self.built:
            self.build(x_train, y_train)
        self._compile(
            learning_rate,
            run_eagerly=run_eagerly,
        )

        # Callbacks
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())
        callbacks.append(pfcallbacks.TerminateOnProtosNaN())

        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=False,
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None,
                embeddings_data=None,
                update_freq='epoch',
            ))
        callbacks.append(
            tf.keras.callbacks.CSVLogger(
                './logs/training.log',
                separator=',',
                append=False,
            ))
        self.history_ = pfcallbacks.LossHistory()
        callbacks.append(self.history_)

        # Train
        history = self.model.fit(x_train,
                                 y_train,
                                 batch_size=batch_size,
                                 callbacks=callbacks,
                                 shuffle=shuffle,
                                 **kwargs)
        return history

    @property
    def loss_history(self):
        """Get the recorded history of losses.

        Returns:
            list: List of losses.
        """
        return self.history_.losses
