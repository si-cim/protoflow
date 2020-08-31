from .competitions import KNNC, WTAC
from .distances import (SED, Distance, EuclideanDistance, LpNormDistance,
                        ManhattanDistance, MatrixDistance,
                        MatrixEuclideanDistance, OmegaDistance,
                        SquaredEuclideanDistance)
from .products import LpSIP
from .prototypes import Prototypes1D

__all__ = [
    'Distance',
    'EuclideanDistance',
    'KNNC',
    'LpNormDistance',
    'LpSIP',
    'ManhattanDistance',
    'MatrixDistance',
    'MatrixEuclideanDistance',
    'OmegaDistance',
    'Prototypes1D',
    'SED',
    'SquaredEuclideanDistance',
    'WTAC',
]
