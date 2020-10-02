"""Module to build, train and evaluate LVQMLN models with Keras."""

from protoflow.applications.deeplvq import DeepLVQ


class LVQMLN(DeepLVQ):
    """Learning Vector Quantization Multi-Layer Network (LVQMLN).

    LVQMLN is a DeepLVQ with a single projection(hidden) layer.

    Arguments:
        mapping_dim (int): Dimension of the space the input is mapped to.

    Keyword Arguments:
        prototypes_per_class (int): Number of prototypes in each class.
            (default 1)
        activation (str): Activation function of the hidden layer.
            (default 'sigmoid')
    """
    def __init__(self,
                 nclasses,
                 input_dim,
                 mapping_dim,
                 activation="sigmoid",
                 **kwargs):
        super().__init__(nclasses,
                         input_dim, [mapping_dim],
                         layer_activations=[activation],
                         **kwargs)
        self.mapping_dim = mapping_dim
