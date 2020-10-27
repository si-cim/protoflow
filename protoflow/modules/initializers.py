"""ProtoFlow initializers."""

import warnings

import numpy as np
import six
import tensorflow as tf
from tensorflow.keras.initializers import *  # noqa: F403, F401
from tensorflow.keras.utils import (deserialize_keras_object,
                                    serialize_keras_object)


class Eye(tf.keras.initializers.Initializer):
    """Initializer that sets the weights to identity matrix(ces)."""
    def __call__(self, shape, dtype="float32"):
        if len(shape) == 2:  # like in global omega for example
            if shape[0] != shape[1]:  # channels first
                raise ValueError(f"Matrix shape {shape} is not square.")
            return tf.eye(shape[0], dtype=dtype)
        elif len(shape) == 3:  # like in local omega for example
            if shape[1] != shape[2]:  # channels first
                raise ValueError(f"Sub-tensors {shape[1]}x{shape[2]} "
                                 "are not square.")
            return tf.stack([tf.eye(shape[1], dtype=dtype)] * shape[0])
        else:
            raise ValueError("Eye initializer can only be used "
                             "for 2D matrices or 3D tensors. "
                             "You requested a {len(shape)}D tensor.")


class _ProtoInit(tf.keras.initializers.Initializer):
    """Base class for ProtoFlow Initializers.

    Use `epsilon` to offset the data if initializing prototypes
    exactly on a data point causes training instability due to
    zero distances.
    """
    def __init__(self,
                 x_train,
                 y_train,
                 prototype_distribution,
                 epsilon=1e-2,
                 **kwargs):
        if not isinstance(x_train, np.ndarray):
            raise TypeError("Expecting a numpy ndarray for "
                            f"inputs `x_train`. "
                            f"You provided: {type(x_train)}.")
        if not isinstance(y_train, np.ndarray):
            raise TypeError("Expecting a numpy ndarray for "
                            f"targets `y_train`. "
                            f"You provided: {type(y_train)}.")
        self.x_train = x_train
        self.y_train = y_train
        self.prototype_distribution = prototype_distribution
        self.epsilon = float(epsilon)
        super().__init__(**kwargs)

    def instantiate(self, dtype="float32"):
        if not isinstance(dtype, str):
            if not isinstance(dtype, np.dtype):
                dtype = dtype.as_numpy_dtype()
        self.unique_labels = np.unique(self.y_train)
        self.prototypes = np.empty(shape=(0, self.x_train.shape[1]),
                                   dtype=dtype)
        self.prototype_labels = np.empty(shape=(0, ), dtype=self.y_train.dtype)

    def validate(self, shape):
        if self.prototypes.shape != shape:
            raise ValueError(f"Expected prototypes of shape: {shape}. "
                             "But the Initializer computed "
                             "prototypes of shape: "
                             f"{self.prototypes.shape}.")


class StratifiedMean(_ProtoInit):
    """Initializer that samples the mean data for each class."""
    def __call__(self, shape, dtype="float32"):
        """Sample and return the mean data for each class."""
        self.instantiate(dtype=dtype)
        for label, num in zip(self.unique_labels, self.prototype_distribution):
            x_label = self.x_train[self.y_train == label]
            x_label_mean = np.mean(x_label, axis=0)
            x_label_mean = x_label_mean.reshape(1, self.x_train.shape[1])
            for _ in range(num):
                self.prototypes = np.append(self.prototypes,
                                            x_label_mean,
                                            axis=0)
            self.prototype_labels = np.append(self.prototype_labels,
                                              [label] * num)
        self.validate(shape=shape)
        return self.prototypes, self.prototype_labels


class StratifiedRandom(_ProtoInit):
    def __call__(self, shape, dtype="float32"):
        """Randomly sample data points depending on their labels."""
        self.instantiate(dtype=dtype)
        for label, num in zip(self.unique_labels, self.prototype_distribution):
            x_where_label = self.x_train[self.y_train == label]
            indices = list(range(x_where_label.shape[0]))
            replace = False
            if num >= len(indices):
                warnings.warn(f"Sampling more prototypes "
                              "than there are data available. "
                              "Are you sure?")
                replace = True
            chosen_indices = np.random.choice(indices,
                                              size=num,
                                              replace=replace)
            chosen_data = x_where_label[chosen_indices]
            random_offset = self.epsilon * np.random.choice(
                [-1, 1], size=chosen_data.shape)
            self.prototypes = np.append(self.prototypes,
                                        chosen_data + random_offset,
                                        axis=0)
            self.prototype_labels = np.append(self.prototype_labels,
                                              [label] * num)
        self.validate(shape=shape)
        return self.prototypes, self.prototype_labels


# Aliases
eye = EYE = Eye
rand = RandomUniform
randn = RandomNormal
stratified_mean = StratifiedMean


def serialize(initializer):
    return serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name="ProtoFlow Initializers")


def get(identifier):
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {"class_name": str(identifier), "config": {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError("Could not interpret initializer identifier: " +
                         str(identifier))
