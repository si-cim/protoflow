"""ProtoFlow layers test suite."""

import unittest

import numpy as np

from protoflow import layers


class TestPrototypes(unittest.TestCase):
    def setUp(self):
        pass

    def test_prototype_labels(self):
        p = layers.Prototypes1D(nclasses=3, prototypes_per_class=1)
        # yapf: disable
        actual = p.prototype_labels.numpy()
        desired = np.array([0, 1, 2])
        # yapf: enable
        self.assertIsNone(np.testing.assert_array_equal(actual, desired))

    def test_prototype_labels_from_pdist(self):
        p = layers.Prototypes1D(nclasses=3, prototype_distribution=[1, 2, 1])
        # yapf: disable
        actual = p.prototype_labels.numpy()
        desired = np.array([0, 1, 1, 2])
        # yapf: enable
        self.assertIsNone(np.testing.assert_array_equal(actual, desired))

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
