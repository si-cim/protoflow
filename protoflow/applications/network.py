"""Module to build, train and evaluate Network models with Keras."""

import numpy as np
import tensorflow as tf

from protoflow.utils import make_directory, accuracy_score


class Network(object):
    """Abstract Class for ProtoFlow Applications.

    Keyword Arguments:
        xtype (str) : Datatype to expect for the inputs `x`.
            (default: 'float')
        ytype (str) : Datatype to expect for the targets `y`.
            (default: 'int')
        dtype (str) : Datatype to use for the network layers.
            (default: 'float32')
    """
    def __init__(self,
                 verbose=True,
                 xtype='float',
                 ytype='int',
                 dtype='float32'):
        super().__init__()
        self.verbose = verbose
        self.xtype = xtype
        self.ytype = ytype
        self.dtype = dtype
        self.built = False

    def _check_validity_of_x(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError('Expecting a numpy ndarray for inputs `x`. '
                            f'You provided: {type(x)}.')
        if x.ndim != 2:
            raise ValueError('Input `x` must be a 2d numpy array. '
                             f'You provided `x` with ndim: {x.ndim}.')
        if x.dtype != self.xtype:
            raise ValueError(f'Input `x` must be of type {self.xtype}. '
                             f'You provided: {x.dtype}.')
        return True

    def _check_validity(self, x, y):
        self._check_validity_of_x(x)
        if not isinstance(y, np.ndarray):
            raise TypeError('Expecting a numpy ndarray for targets `y`. '
                            f'You provided: {type(y)}.')
        if x.shape[0] != y.shape[0]:
            raise ValueError('Not enough or too many target instances. '
                             f'`x.shape`: {x.shape} does not match '
                             f'`y.shape`: {y.shape} in axis 0.')
        if y.ndim != 1:
            raise ValueError('Targets `y` must be a 1d array. '
                             f'You provided `y` with ndim: {y.ndim}.')
        if y.dtype != self.ytype:
            raise ValueError(f'Input `y` must be of type {self.ytype}. '
                             f'You provided: {y.dtype}.')
        return True

    def _check_model_availability(self):
        if not hasattr(self, 'model'):
            raise AttributeError('`self.model` is not available. '
                                 'Either build the model manually '
                                 'with `self.build()` or call `self.fit()` '
                                 'on the training data.')

    def build(self):
        """Build the Application.

        Important:
            Overwrite this method in your Application.
        """
        raise NotImplementedError('Please implement this method!')

    def compile(self):
        """Compile the Application.

        Important:
            Overwrite this method in your Application.
        """
        raise NotImplementedError('Please implement this method!')

    def predict(self, inputs):
        """Convenience method. Returns `self.model.predict(inputs)`."""
        self._check_model_availability()
        return self.model.predict(inputs)

    def summary(self):
        """Convenience method. Calls `self.model.summary()`."""
        self._check_model_availability()
        return self.model.summary()

    def score(self, x_test, y_test, verbose=False):
        """Default score method. Calculate accuracy for test dataset.

        Args:
            x_test: Test inputs.
            y_test: True targets.

        Keyword Arguments:
            verbose (bool): Print Accuracy after calculation.
        """
        self._check_validity(x_test, y_test)
        y_pred = self.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        if verbose:
            print(f'Accuracy score: {acc * 100:02.03f}%')
        return acc

    def save_weights_as_numpy(self,
                              save_dir='./saved_weights',
                              prefix='weights'):
        """Save weights of every layer as numpy formated file.

        Keyword Arguments:
            save_dir: Folder to save the files to.
            prefix: Filename prefix.
        """
        make_directory(save_dir)
        for l in self.layers:
            for i, w in enumerate(l.get_weights()):
                np.save(f'{save_dir}/{prefix}_{l.name}_w{i}.npy', w)

    def load_weights_from_numpy(self,
                                save_dir='./saved_weights',
                                prefix='weights'):
        """Save weights of every layer as numpy formated File.

        Keyword Arguments:
            save_dir: Folder to read the files from.
            prefix: Filename prefix.
        """
        for l in self.layers:
            weights = []
            for i, w_ in enumerate(l.get_weights()):
                fname = f'{save_dir}/{prefix}_{l.name}_w{i}.npy'
                w = np.load(fname)
                if w.shape != w_.shape:
                    raise ValueError(f'Shape mismatch when loading {fname}')
                weights.append(w)
            l.set_weights(weights)
        print('Successfully loaded all weights.')

    def save_weights_as_mat(self, save_dir='./saved_weights', name='weights'):
        """Save weights of every layer as numpy formated file.

        Keyword Arguments:
            save_dir: Folder to save the weights to.
            prefix: Filename.
        """
        import scipyio
        make_directory(save_dir)
        weights = {}
        for l in self.model.layers:
            for i, w in enumerate(l.get_weights()):
                weights[f'{l.name}_w{i}'] = w
        scipyio.savemat(f'{save_dir}/{name}.mat', weights)

    def get_config(self):
        import copy
        layer_configs = []
        for layer in self.model.layers:
            layer_configs.append({
                'class_name': layer.__class__.__name__,
                'config': layer.get_config()
            })
        config = {
            'name': self.model.name,
            'layers': copy.deepcopy(layer_configs)
        }
        if self.model._build_input_shape:
            config['build_input_shape'] = self.model._build_input_shape
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if 'name' in config:
            name = config['name']
            build_input_shape = config.get('build_input_shape')
            layer_configs = config['layers']
        else:  # legacy config file
            name = build_input_shape = None
            layer_configs = config
        model = cls(name=name)
        for conf in layer_configs:
            layer = tf.keras.layers.deserialize(conf,
                                                custom_objects=custom_objects)
            model.add(layer)
        if not model.inputs and build_input_shape:
            model.build(build_input_shape)
        return model
