"""Module to build, train and evaluate LVQMLN models with Keras."""

from protoflow.applications.deeplvq import DeepLVQ


class LVQMLN(DeepLVQ):
    """Learning Vector Quantization Multi-Layer Network (LVQMLN).

    LVQMLN is a DeepLVQ with a single projection(hidden) layer.

    Arguments:
        projection_dim (int): Dimension of the space the input is projected to.

    Keyword Arguments:
        prototypes_per_class (int): Number of prototypes in each class.
            (default 1)
        activation (str): Activation function of the hidden layer.
            (default 'sigmoid')
    """
    def __init__(self,
                 projection_dim,
                 prototypes_per_class=1,
                 activation='sigmoid',
                 **kwargs):
        super().__init__([projection_dim],
                         prototypes_per_class,
                         layer_activations=[activation],
                         **kwargs)
        self.projection_dim = projection_dim

    @property
    def omega(self):
        """Get the weights of the projection layer.

        Returns:
            numpy.ndarray: projection weights.
        """
        omega = self.projections[0].get_weights()[0]
        return omega
