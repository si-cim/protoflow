"""ProtoFlow datasets test suite."""

import unittest

import numpy as np

import protoflow as pf


class TestDatasets(unittest.TestCase):
    def setUp(self):
        pass

    def test_tecator(self):
        (x_train, y_train) = pf.datasets.tecator.load_data()
        self.assertTrue(isinstance(x_train, np.ndarray))
        self.assertTrue(isinstance(y_train, np.ndarray))

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
