"""ProtoFlow layers test suite."""

import io
import unittest

import numpy as np
from protoflow import applications


class TestNetwork(unittest.TestCase):
    def setUp(self):
        pass

    def test_object_init(self):
        net = applications.Network(verbose=False)
        self.assertFalse(net.built)
        self.assertFalse(net.verbose)

    def test_model_availability_errors(self):
        net = applications.Network()
        with self.assertRaises(AttributeError):
            net.summary()

    def tearDown(self):
        pass


class TestKNN(unittest.TestCase):
    def setUp(self):
        self.x = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype='float')
        self.y = np.array([
            0,
            0,
            0,
            1,
        ], dtype='int')

    def test_object_init(self):
        clf = applications.KNN(k=99)
        self.assertEqual(clf.k, 99)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_summary(self, stdout):
        clf = applications.KNN(k=1)
        clf.build(self.x, self.y)
        clf.summary()

    def test_fit(self):
        clf = applications.KNN(k=1)
        clf.fit(self.x, self.y)

    def test_predict_k_1(self):
        clf = applications.KNN(k=1)
        clf.fit(self.x, self.y)
        y_pred = clf.predict(self.x)
        self.assertIsNone(
            np.testing.assert_array_almost_equal(y_pred, self.y, decimal=5))

    def tearDown(self):
        del self.x
        del self.y


class TestGLVQ(unittest.TestCase):
    def setUp(self):
        # yapf: disable
        self.x = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1],
                           [2, 0],
                           [0, 2],
                           [2, 2]], dtype='float')
        self.y = np.array([0,
                           0,
                           0,
                           0,
                           1,
                           1,
                           1], dtype='int')
        # yapf: enable

    def test_object_init(self):
        clf = applications.GLVQ(2)
        self.assertEqual(clf.prototypes_per_class, 2)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_summary(self, stdout):
        clf = applications.GLVQ(5)
        clf.build(self.x, self.y, prototype_initializer='rand')
        clf.summary()

    def test_fit(self):
        clf = applications.GLVQ(1)
        clf.fit(self.x, self.y)

    def tearDown(self):
        del self.x
        del self.y


class TestGMLVQ(unittest.TestCase):
    def setUp(self):
        # yapf: disable
        self.x = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1],
                           [2, 0],
                           [0, 2],
                           [2, 2]], dtype='float')
        self.y = np.array([0,
                           0,
                           0,
                           0,
                           1,
                           1,
                           1], dtype='int')
        # yapf: enable

    def test_object_init(self):
        clf = applications.GMLVQ(2)
        self.assertEqual(clf.prototypes_per_class, 2)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_summary(self, stdout):
        clf = applications.GMLVQ(5)
        clf.build(self.x,
                  self.y,
                  prototype_initializer='rand',
                  matrix_initializer='eye')
        clf.summary()

    def test_fit(self):
        clf = applications.GMLVQ(1)
        clf.fit(self.x, self.y)

    def tearDown(self):
        del self.x
        del self.y


class TestLVQMLN(unittest.TestCase):
    def setUp(self):
        # yapf: disable
        self.x = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1],
                           [2, 0],
                           [0, 2],
                           [2, 2]], dtype='float')
        self.y = np.array([0,
                           0,
                           0,
                           0,
                           1,
                           1,
                           1], dtype='int')
        # yapf: enable

    def test_object_init(self):
        clf = applications.LVQMLN(1, 1, activation='swish_beta')
        self.assertEqual(clf.prototypes_per_class, 1)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_summary(self, stdout):
        clf = applications.LVQMLN(5, 2)
        clf.build(self.x, self.y, prototype_initializer='rand')
        clf.summary()

    def test_fit(self):
        clf = applications.LVQMLN(1, 3)
        clf.fit(self.x, self.y)

    def tearDown(self):
        del self.x
        del self.y


class TestDeepLVQ(unittest.TestCase):
    def setUp(self):
        # yapf: disable
        self.x = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1],
                           [2, 0],
                           [0, 2],
                           [2, 2]], dtype='float')
        self.y = np.array([0,
                           0,
                           0,
                           0,
                           1,
                           1,
                           1], dtype='int')
        # yapf: enable

    def test_object_init(self):
        clf = applications.DeepLVQ([1], 1, layer_activations=['swish_beta'])
        self.assertEqual(clf.prototypes_per_class, 1)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_summary(self, stdout):
        clf = applications.DeepLVQ([100, 10],
                                   2,
                                   layer_activations=['sigmoid', 'swish_beta'],
                                   layer_biases=[True, False])
        clf.build(self.x, self.y, prototype_initializer='rand')
        clf.summary()

    def test_fit(self):
        clf = applications.DeepLVQ(
            [100, 10, 3],
            1,
            layer_activations=['sigmoid', 'swish_beta', 'sigmoid'],
            layer_biases=[True, False, False])
        clf.fit(self.x, self.y)

    def tearDown(self):
        del self.x
        del self.y


if __name__ == '__main__':
    unittest.main()
