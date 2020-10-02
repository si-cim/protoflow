"""ProtoFlow layers test suite."""

import io
import unittest

import numpy as np

import protoflow as pf


class TestGLVQ(unittest.TestCase):
    def setUp(self):
        ndata = 100
        nclasses = 5
        input_dim = 10
        self.model = pf.applications.GLVQ(nclasses=nclasses,
                                          input_dim=input_dim,
                                          prototypes_per_class=3)
        self.x = np.random.rand(ndata, input_dim)
        self.y = np.random.randint(0, nclasses, size=(ndata, ))

    def test_prototype_distribution(self):
        self.assertEqual(self.model.prototype_layer.prototype_distribution,
                         [3, 3, 3, 3, 3])

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_summary(self, stdout):
        self.model.summary()
        summary_string = stdout.getvalue()
        stdout.close()
        self.assertIn("Trainable params:", summary_string)

    def test_compile_and_fit(self):
        self.model.compile(optimizer='adam')
        self.model.fit(self.x, self.y, batch_size=64)

    def tearDown(self):
        del self.model
        del self.x
        del self.y


class TestGMLVQ(unittest.TestCase):
    def setUp(self):
        ndata = 100
        nclasses = 5
        input_dim = 10
        mapping_dim = 2
        self.model = pf.applications.GMLVQ(nclasses=nclasses,
                                           input_dim=input_dim,
                                           mapping_dim=mapping_dim,
                                           prototypes_per_class=3)
        self.x = np.random.rand(ndata, input_dim)
        self.y = np.random.randint(0, nclasses, size=(ndata, ))

    def test_prototype_distribution(self):
        self.assertEqual(self.model.prototype_layer.prototype_distribution,
                         [3, 3, 3, 3, 3])

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_summary(self, stdout):
        self.model.summary()
        summary_string = stdout.getvalue()
        stdout.close()
        self.assertIn("Trainable params:", summary_string)

    def test_compile_and_fit(self):
        self.model.compile(optimizer='adam')
        self.model.fit(self.x, self.y, batch_size=64)

    def tearDown(self):
        del self.model
        del self.x
        del self.y


class TestDeepLVQ(unittest.TestCase):
    def setUp(self):
        ndata = 100
        nclasses = 5
        input_dim = 10
        self.model = pf.applications.DeepLVQ(nclasses=nclasses,
                                             input_dim=input_dim,
                                             hidden_units=[1, 2],
                                             prototypes_per_class=3)
        self.x = np.random.rand(ndata, input_dim)
        self.y = np.random.randint(0, nclasses, size=(ndata, ))

    def test_prototype_distribution(self):
        self.assertEqual(self.model.prototype_layer.prototype_distribution,
                         [3, 3, 3, 3, 3])

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_summary(self, stdout):
        self.model.summary()
        summary_string = stdout.getvalue()
        stdout.close()
        self.assertIn("Trainable params:", summary_string)

    def test_compile_and_fit(self):
        self.model.compile(optimizer='adam')
        self.model.fit(self.x, self.y, batch_size=64)

    def tearDown(self):
        del self.model
        del self.x
        del self.y


if __name__ == '__main__':
    unittest.main()
