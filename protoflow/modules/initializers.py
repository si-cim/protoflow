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
    def __call__(self, shape, dtype='float32'):
        if len(shape) == 2:  # like in global omega for example
            if shape[0] != shape[1]:  # channels first
                raise ValueError(f'Matrix shape {shape} is not square.')
            return tf.eye(shape[0], dtype=dtype)
        elif len(shape) == 3:  # like in local omega for example
            if shape[1] != shape[2]:  # channels first
                raise ValueError(f'Sub-tensors {shape[1]}x{shape[2]} '
                                 'are not square.')
            return tf.stack([tf.eye(shape[1], dtype=dtype)] * shape[0])
        else:
            raise ValueError('Eye initializer can only be used '
                             'for 2D matrices or 3D tensors. '
                             'You requested a {len(shape)}D tensor.')


def get_classwise_random_initializer(x_train,
                                     y_train,
                                     prototype_distribution,
                                     verbose=False):
    class ClasswiseRandom(tf.keras.initializers.Initializer):
        """Keras Initializer that randomly samples the inputs."""
        def __init__(self,
                     x_train,
                     y_train,
                     prototype_distribution,
                     verbose=verbose):
            if not isinstance(x_train, np.ndarray):
                raise TypeError('Expecting a numpy ndarray for '
                                f'inputs `x_train`. '
                                f'You provided: {type(x_train)}.')
            if not isinstance(y_train, np.ndarray):
                raise TypeError('Expecting a numpy ndarray for '
                                f'targets `y_train`. '
                                f'You provided: {type(y_train)}.')
            self.x_train = x_train
            self.y_train = y_train
            self.prototype_distribution = prototype_distribution
            self.verbose = verbose

        def __call__(self, shape, dtype='float32'):
            """Sample data means by class."""
            if not isinstance(dtype, str):
                if not isinstance(dtype, np.dtype):
                    dtype = dtype.as_numpy_dtype()
            labels = np.unique(self.y_train)  # class labels
            self.prototypes = np.empty(shape=(0, self.x_train.shape[1]),
                                       dtype=dtype)
            self.prototype_labels = np.empty(shape=(0, ), dtype=y_train.dtype)
            for label, num in zip(labels, self.prototype_distribution):
                x_label = self.x_train[self.y_train == label]
                if self.verbose:
                    print(f'Found {x_label.shape[0]} samples with '
                          f'class {label}. Selecting {num} copies '
                          f'of the mean of class {label}.')
                indices = list(range(x_label.shape[0]))
                replace = False
                if num >= len(indices):
                    warnings.warn(f'Randomly sampling more prototypes '
                                  'than there are data available.')
                    replace = True
                chosen_indices = np.random.choice(indices,
                                                  size=num,
                                                  replace=replace)
                self.prototypes = np.append(self.prototypes,
                                            x_label[chosen_indices],
                                            axis=0)
                self.prototype_labels = np.append(self.prototype_labels,
                                                  [label] * num)

            if self.prototypes.shape != shape:
                raise ValueError(f'Layer expected weights of shape: {shape}. '
                                 'But ClasswiseMean Initializer computed '
                                 'prototypes of shape: '
                                 f'{self.prototypes.shape}.')
            return self.prototypes

    return ClasswiseRandom(x_train, y_train, prototype_distribution, verbose)


def get_classwise_mean_initializer(x_train,
                                   y_train,
                                   prototype_distribution,
                                   verbose=False):
    class ClasswiseMean(tf.keras.initializers.Initializer):
        """Keras Initializer that samples the classwise mean."""
        def __init__(self,
                     x_train,
                     y_train,
                     prototype_distribution,
                     verbose=verbose):
            if not isinstance(x_train, np.ndarray):
                raise TypeError('Expecting a numpy ndarray for '
                                f'inputs `x_train`. '
                                f'You provided: {type(x_train)}.')
            if not isinstance(y_train, np.ndarray):
                raise TypeError('Expecting a numpy ndarray for '
                                f'targets `y_train`. '
                                f'You provided: {type(y_train)}.')
            self.x_train = x_train
            self.y_train = y_train
            self.prototype_distribution = prototype_distribution
            self.verbose = verbose

        def __call__(self, shape, dtype='float32'):
            """Sample data means by class."""
            if not isinstance(dtype, str):
                if not isinstance(dtype, np.dtype):
                    dtype = dtype.as_numpy_dtype()
            labels = np.unique(self.y_train)  # class labels
            self.prototypes = np.empty(shape=(0, self.x_train.shape[1]),
                                       dtype=dtype)
            self.prototype_labels = np.empty(shape=(0, ), dtype=y_train.dtype)
            for label, num in zip(labels, self.prototype_distribution):
                x_label = self.x_train[self.y_train == label]
                if self.verbose:
                    print(f'Found {x_label.shape[0]} samples with '
                          f'class {label}. Selecting {num} copies '
                          f'of the mean of class {label}.')
                x_label_mean = np.mean(x_label, axis=0)
                x_label_mean = x_label_mean.reshape(1, self.x_train.shape[1])
                for _ in range(num):
                    self.prototypes = np.append(self.prototypes,
                                                x_label_mean,
                                                axis=0)
                self.prototype_labels = np.append(self.prototype_labels,
                                                  [label] * num)
            if self.prototypes.shape != shape:
                raise ValueError(f'Layer expected weights of shape: {shape}. '
                                 'But ClasswiseMean Initializer computed '
                                 'prototypes of shape: '
                                 f'{self.prototypes.shape}.')
            return self.prototypes

    return ClasswiseMean(x_train, y_train, prototype_distribution, verbose)


PROTOTYPE_INITIALIZERS = {
    'mean': get_classwise_mean_initializer,
    'rand': get_classwise_random_initializer,
    # Aliases
    'random': get_classwise_random_initializer,
    'classmean': get_classwise_random_initializer,
}

# Aliases
eye = EYE = Eye
rand = RandomUniform
randn = RandomNormal


def serialize(initializer):
    return serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='ProtoFlow initializer')


def get(identifier):
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret initializer identifier: ' +
                         str(identifier))
