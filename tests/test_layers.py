"""ProtoFlow layers test suite."""

import unittest

import numpy as np

from protoflow import layers


class TestDistances(unittest.TestCase):
    def setUp(self):
        pass

    def test_manhattan(self):
        d = layers.ManhattanDistance(num_of_prototypes=3,
                                     prototype_dim=2,
                                     prototype_labels=[0, 1, 2],
                                     prototype_initializer='ones')
        # yapf: disable
        x = np.array([[0, 0],
                      [1, 1]], dtype='float32')
        actual = d(x)
        desired = np.array([[2.0000, 2.0000, 2.0000],
                            [0.0000, 0.0000, 0.0000]])
        # yapf: enable
        self.assertIsNone(
            np.testing.assert_array_almost_equal(actual, desired, decimal=5))

    def test_euclidean(self):
        d = layers.EuclideanDistance(num_of_prototypes=3,
                                     prototype_dim=2,
                                     prototype_labels=[0, 1, 2],
                                     prototype_initializer='ones')
        # yapf: disable
        x = np.array([[0, 0],
                      [1, 1]], dtype='float32')
        actual = d(x)
        desired = np.array([[1.4142, 1.4142, 1.4142],
                            [0.0000, 0.0000, 0.0000]])
        # yapf: enable
        self.assertIsNone(
            np.testing.assert_array_almost_equal(actual, desired, decimal=5))

    def test_squared_euclidean(self):
        d = layers.SquaredEuclideanDistance(num_of_prototypes=3,
                                            prototype_dim=2,
                                            prototype_labels=[0, 1, 2],
                                            prototype_initializer='ones')
        # yapf: disable
        x = np.array([[0, 0],
                      [1, 1]], dtype='float32')
        actual = d(x)
        desired = np.array([[1.4142, 1.4142, 1.4142],
                            [0.0000, 0.0000, 0.0000]]) ** 2
        # yapf: enable
        self.assertIsNone(
            np.testing.assert_array_almost_equal(actual, desired, decimal=4))

    def tearDown(self):
        pass


class TestCompetitions(unittest.TestCase):
    def setUp(self):
        pass

    def test_wtac(self):
        c = layers.WTAC(prototype_labels=[0, 1, 2], )
        # yapf: disable
        x = np.array([[0.1, 0.2, 0.3],
                      [0.2, 0.1, 0.3],
                      [0.3, 0.1, 0.2],
                      [0.3, 0.2, 0.1]], dtype='float32')
        actual = c(x).numpy()
        desired = np.array([0, 1, 1, 2])
        # yapf: enable
        self.assertIsNone(np.testing.assert_array_equal(actual, desired))

    def test_wtac_one_proto_per_class(self):
        c = layers.WTAC(prototype_labels=[0, 1, 2], )
        # yapf: disable
        x = np.array([[0.1, 0.2, 0.3],
                      [0.2, 0.1, 0.3],
                      [0.3, 0.1, 0.2],
                      [0.3, 0.2, 0.1]], dtype='float32')
        actual = c(x).numpy()
        desired = np.array([0, 1, 1, 2])
        # yapf: enable
        self.assertIsNone(np.testing.assert_array_equal(actual, desired))

    def test_wtac_two_proto_per_class(self):
        prototype_labels = [0, 1, 2, 0, 1, 2]
        c = layers.WTAC(prototype_labels=prototype_labels, )
        # yapf: disable
        x = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                      [0.4, 0.5, 0.1, 0.2, 0.6, 0.3],
                      [0.4, 0.3, 0.1, 0.5, 0.6, 0.2],
                      [0.6, 0.3, 0.2, 0.5, 0.1, 0.4]], dtype='float32')
        actual = c(x).numpy()
        desired = []
        for i in np.argmin(x, axis=1):
            desired.append(prototype_labels[i])
        desired = np.array(desired)  # [0, 2, 2, 1]
        # yapf: enable
        self.assertIsNone(np.testing.assert_array_equal(actual, desired))

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
