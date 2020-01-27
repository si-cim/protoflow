"""ProtoFlow modules test suite."""

import unittest

import numpy as np

from protoflow.modules import initializers, losses


class TestLosses(unittest.TestCase):
    def setUp(self):
        pass

    def test_glvq_identity_squashing_deserialization(self):
        losses.GLVQLoss(prototype_labels=[0, 1, 2], squashing='identity')

    def test_glvq_linear_squashing_deserialization(self):
        losses.GLVQLoss(prototype_labels=[0, 1, 2], squashing='linear')

    def test_glvq_sigmoid_squashing_deserialization(self):
        losses.GLVQLoss(prototype_labels=[0, 1, 2], squashing='sigmoid')

    def test_glvq_swish_squashing_deserialization(self):
        losses.GLVQLoss(prototype_labels=[0, 1, 2],
                        squashing='swish',
                        beta=0.1)

    def test_glvq_relu_squashing_deserialization(self):
        losses.GLVQLoss(prototype_labels=[0, 1, 2], squashing='relu', beta=0.1)

    def test_glvq_softplus_squashing_deserialization(self):
        losses.GLVQLoss(prototype_labels=[0, 1, 2],
                        squashing='softplus',
                        beta=0.1)

    def test_glvq_foobar_squashing_deserialization(self):
        with self.assertRaises(ValueError):
            losses.GLVQLoss(prototype_labels=[0, 1, 2], squashing='foobar')

    def test_glvq_blubb_squashing_deserialization(self):
        with self.assertRaises(ValueError):
            losses.GLVQLoss(prototype_labels=[0, 1, 2], squashing='blubb')

    def test_glvq_identity_squashing_serialization(self):
        losses.GLVQLoss(prototype_labels=[0, 1, 2],
                        squashing='identity').get_config()

    def test_glvq_linear_squashing_serialization(self):
        losses.GLVQLoss(prototype_labels=[0, 1, 2],
                        squashing='linear').get_config()

    def test_glvq_sigmoid_squashing_serialization(self):
        losses.GLVQLoss(prototype_labels=[0, 1, 2],
                        squashing='sigmoid').get_config()

    def test_glvq_swish_squashing_serialization(self):
        losses.GLVQLoss(prototype_labels=[0, 1, 2],
                        squashing='swish',
                        beta=0.1).get_config()

    def test_glvq_relu_squashing_serialization(self):
        losses.GLVQLoss(prototype_labels=[0, 1, 2], squashing='relu',
                        beta=0.1).get_config()

    def test_glvq_softplus_squashing_serialization(self):
        losses.GLVQLoss(prototype_labels=[0, 1, 2],
                        squashing='softplus',
                        beta=0.1).get_config()

    def tearDown(self):
        pass


class TestInitializers(unittest.TestCase):
    def setUp(self):
        pass

    def test_eye_2d_n2(self):
        eye = initializers.Eye()
        # yapf: disable
        desired = np.array([[1, 0],
                            [0, 1]], dtype='float32')
        actual = eye(shape=(2, 2))
        # yapf: enable
        self.assertIsNone(np.testing.assert_array_almost_equal(
            actual, desired))

    def test_eye_2d_n3(self):
        eye = initializers.Eye()
        # yapf: disable
        desired = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]], dtype='float32')
        actual = eye(shape=(3, 3))
        # yapf: enable
        self.assertIsNone(np.testing.assert_array_almost_equal(
            actual, desired))

    def test_eye_2d_n4(self):
        eye = initializers.Eye()
        # yapf: disable
        desired = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype='float32')
        actual = eye(shape=(4, 4))
        # yapf: enable
        self.assertIsNone(np.testing.assert_array_almost_equal(
            actual, desired))

    def test_eye_3d_n4_2(self):
        eye = initializers.Eye()
        # yapf: disable
        desired = np.array([[[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]],

                            [[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]]], dtype='float32')
        actual = eye(shape=(2, 4, 4))
        # yapf: enable
        self.assertIsNone(np.testing.assert_array_almost_equal(
            actual, desired))

    def test_eye_2d_rectangular(self):
        with self.assertRaises(ValueError):
            eye = initializers.Eye()
            _ = eye(shape=(2, 3))

    def test_eye_dtype_float32(self):
        eye = initializers.Eye()
        desired = 'float32'
        actual = eye(shape=(4, 4), dtype=desired).dtype
        self.assertEqual(actual, desired)

    def test_eye_dtype_float64(self):
        eye = initializers.Eye()
        desired = 'float64'
        actual = eye(shape=(4, 4), dtype=desired).dtype
        self.assertEqual(actual, desired)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
