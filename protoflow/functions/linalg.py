"""ProtoFlow normalization functions."""

import tensorflow as tf
from tensorflow.keras import backend as K


def det(tensors, keepdims=False):
    """Determinant with keepdims."""
    d = tf.linalg.det(tensors)
    if keepdims:
        d = tf.expand_dims(tf.expand_dims(d, -1), -1)

    return d


def trace(tensors, keepdims=False):
    """Compute the trace of a squared matrix."""
    def mixed_shape(inputs):
        if not tf.is_tensor(inputs):
            raise ValueError('Input must be a tensor.')
        with K.name_scope('mixed_shape'):
            int_shape = list(K.int_shape(inputs))
            # sometimes int_shape returns mixed integer types
            int_shape = [int(i) if i is not None else i for i in int_shape]
            tensor_shape = K.shape(inputs)

            for i, s in enumerate(int_shape):
                if s is None:
                    int_shape[i] = tensor_shape[i]
            return tuple(int_shape)

    def equal_int_shape(shape_1, shape_2):
        if not isinstance(shape_1, (tuple, list)) or not isinstance(
                shape_2, (tuple, list)):
            raise ValueError('Input shapes must list or tuple.')
        for shape in [shape_1, shape_2]:
            if not all([isinstance(x, int) or x is None for x in shape]):
                raise ValueError('Input shapes must be list or tuple '
                                 'of int and None values.')

        if len(shape_1) != len(shape_2):
            return False
        else:
            for axis, value in enumerate(shape_1):
                if value is not None and shape_2[axis] not in {value, None}:
                    return False
            return True

    with K.name_scope('trace'):
        shape = mixed_shape(tensors)
        int_shape = K.int_shape(tensors)

        if not equal_int_shape([int_shape[-1]], [int_shape[-2]]):
            raise ValueError(
                'The matrix dimension (the last two dimensions) of the '
                f'tensor must be square. You provide: {int_shape[-2:]}.')
        if int_shape[-1] is None and int_shape[-2] is None:
            raise ValueError('At least one dimension of the matrix '
                             f'must be defined. You provide: {int_shape}')

        # K.eye() doesn't accept placeholders. Thus, one dim must be specified.
        if int_shape[-1] is None:
            matrix_dim = shape[-2]
        else:
            matrix_dim = shape[-1]

        t = K.sum(tensors * K.eye(matrix_dim), axis=[-1, -2])
        if keepdims:
            t = K.expand_dims(K.expand_dims(t, -1), -1)

        return t
