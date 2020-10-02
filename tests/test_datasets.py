"""ProtoFlow datasets test suite."""

import unittest

import numpy as np

import protoflow as pf


class TestDatasets(unittest.TestCase):
    def setUp(self):
        pass

    def check_if_numpy(self, arrays):
        for array in arrays:
            self.assertTrue(isinstance(array, np.ndarray))

    # def test_flc1(self):
    #     train, test = pf.datasets.flc1.load_data()
    #     self.check_if_numpy(arrays=train + test)

    # def test_tecator(self):
    #     train, test = pf.datasets.tecator.load_data()
    #     self.check_if_numpy(arrays=train + test)

    # def test_whisky(self):
    #     arrays = pf.datasets.whisky.load_data()
    #     self.check_if_numpy(arrays)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
