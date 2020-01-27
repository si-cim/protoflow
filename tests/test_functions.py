"""ProtoFlow functions test suite."""

import unittest

import numpy as np

from protoflow.functions import distances


class TestDistances(unittest.TestCase):
    def setUp(self):
        pass

    def test_lpnorm_p2_1d(self):
        # yapf: disable
        x = np.array([[0, 0]], dtype='float32')
        w = np.array([[1, 1]], dtype='float32')
        actual = distances.lpnorm_distance(x, w, p=2)
        desired = np.array([[1.4142]])
        # yapf: enable
        self.assertIsNone(
            np.testing.assert_array_almost_equal(actual, desired, decimal=4))

    def test_lpnorm_p2_2d(self):
        # yapf: disable
        x = np.array([[0, 0],
                      [1, 1]], dtype='float32')
        w = np.array([[1, 1],
                      [1, 1],
                      [1, 1]], dtype='float32')
        actual = distances.lpnorm_distance(x, w, p=2)
        desired = np.array([[1.4142, 1.4142, 1.4142],
                            [0.0000, 0.0000, 0.0000]])
        # yapf: enable
        self.assertIsNone(
            np.testing.assert_array_almost_equal(actual, desired, decimal=4))

    def test_omega(self):
        # yapf: disable
        x = np.array([[0, 0],
                      [1, 1]], dtype='float32')
        w = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]], dtype='float32')
        omega = np.eye(w.shape[1], dtype='float32')
        actual = distances.omega_distance(x, w, omega)
        desired = np.array([[0.0000, 1.0000, 1.0000, 1.4142],
                            [1.4142, 1.0000, 1.0000, 0.0000]]) ** 2
        # yapf: enable
        self.assertIsNone(
            np.testing.assert_array_almost_equal(actual, desired, decimal=4))

    def test_lomega_eye_omegas(self):
        # yapf: disable
        x = np.array([[0, 0],
                      [1, 1]], dtype='float32')
        w = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]], dtype='float32')
        omegas = np.stack([np.eye(w.shape[1], dtype='float32')] * w.shape[0])
        actual = distances.lomega_distance(x, w, omegas)
        desired = np.array([[0.0000, 1.0000, 1.0000, 1.4142],
                            [1.4142, 1.0000, 1.0000, 0.0000]]) ** 2
        # yapf: enable
        self.assertIsNone(
            np.testing.assert_array_almost_equal(actual, desired, decimal=4))

    def test_lomega_zeros_omegas(self):
        # yapf: disable
        x = np.array([[0, 0],
                      [1, 1]], dtype='float32')
        w = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]], dtype='float32')
        omegas = np.stack([np.zeros(
            (w.shape[1], w.shape[1]), dtype='float32')] * w.shape[0])
        actual = distances.lomega_distance(x, w, omegas)
        desired = np.array([[0.000000, 0.000000, 0.000000, 0.000000],
                            [0.000000, 0.000000, 0.000000, 0.000000]])
        # yapf: enable
        self.assertIsNone(
            np.testing.assert_array_almost_equal(actual, desired, decimal=4))

    def test_lomega_ones_omegas(self):
        # yapf: disable
        x = np.array([[0, 0],
                      [1, 1]], dtype='float32')
        w = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]], dtype='float32')
        omegas = np.stack([np.ones(
            (w.shape[1], w.shape[1]), dtype='float32')] * w.shape[0])
        actual = distances.lomega_distance(x, w, omegas)
        desired = np.array([[0.000000, 1.4142135, 1.4142135, 2.828427],
                            [2.828427, 1.4142135, 1.4142135, 0.000000]]) ** 2
        # yapf: enable
        self.assertIsNone(
            np.testing.assert_array_almost_equal(actual, desired, decimal=4))


if __name__ == '__main__':
    unittest.main()
