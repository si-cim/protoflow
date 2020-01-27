"""ProtoFlow normalization functions."""

import tensorflow as tf
from tensorflow.keras import backend as K

from .linalg import trace


def orthogonalization(tensors):
    """Perform orthogonalization via polar decomposition."""
    def mixed_shape(inputs):
        if not K.is_tensor(inputs):
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

    with K.name_scope('orthogonalization'):
        _, u, v = tf.svd(tensors, full_matrices=False, compute_uv=True)
        u_shape = mixed_shape(u)
        v_shape = mixed_shape(v)

        # reshape to (num x N x M)
        u = K.reshape(u, (-1, u_shape[-2], u_shape[-1]))
        v = K.reshape(v, (-1, v_shape[-2], v_shape[-1]))

        out = K.batch_dot(u, K.permute_dimensions(v, [0, 2, 1]))

        out = K.reshape(out, u_shape[:-1] + (v_shape[-2], ))

        return out


def trace_normalization(tensors, epsilon=K.epsilon()):
    with K.name_scope('trace_normalization'):
        constant = trace(tensors, keepdims=True)

        if epsilon != 0:
            constant = K.maximum(constant, epsilon)

        return tensors / constant


def omega_normalization(tensors, epsilon=K.epsilon()):
    with K.name_scope('omega_normalization'):
        ndim = K.ndim(tensors)

        # batch matrices
        if ndim >= 3:
            axes = ndim - 1
            s_tensors = K.batch_dot(tensors, tensors, [axes, axes])
        # non-batch
        else:
            s_tensors = K.dot(tensors, K.transpose(tensors))

        t = trace(s_tensors, keepdims=True)
        if epsilon == 0:
            constant = K.sqrt(t)
        else:
            constant = K.sqrt(K.maximum(t, epsilon))

        return tensors / constant
