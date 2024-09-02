import unittest
from unittest.mock import patch
import torch as pt
from swak.pt.mix import StackWeightedSumMixer


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.mix = StackWeightedSumMixer(3)

    def test_has_n_features(self):
        self.assertTrue(hasattr(self.mix, 'n_features'))

    def test_n_features(self):
        self.assertIsInstance(self.mix.n_features, int)
        self.assertEqual(3, self.mix.n_features)

    def test_has_coefficients(self):
        self.assertTrue(hasattr(self.mix, '_coefficients'))

    def test_coefficients(self):
        self.assertIsInstance(self.mix._coefficients, pt.nn.Parameter)
        pt.testing.assert_close(self.mix._coefficients, pt.ones(3))

    def test_has_importance(self):
        self.assertTrue(hasattr(self.mix, 'importance'))

    def test_importance(self):
        expected = pt.tensor([1/3, 1/3, 1/3])
        pt.testing.assert_close(self.mix.importance, expected)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.mix, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.mix.reset_parameters))

    @patch('torch.ones', return_value=pt.tensor([0., 0.5, 1.0]))
    def test_reset_parameters_called(self, _):
        self.mix.reset_parameters()
        expected = pt.nn.Softmax(0)(pt.tensor([0., 0.5, 1.0]))
        pt.testing.assert_close(self.mix.importance, expected)

    def test_has_new(self):
        self.assertTrue(hasattr(self.mix, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.mix.new))

    def test_call_new_default(self):
        new = self.mix.new()
        self.assertIsInstance(new, StackWeightedSumMixer)
        self.assertEqual(self.mix.n_features, new.n_features)

    def test_call_new_update(self):
        new = self.mix.new(4)
        self.assertIsInstance(new, StackWeightedSumMixer)
        self.assertEqual(4, new.n_features)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mix = StackWeightedSumMixer(2)
        f1 = pt.tensor([2., 3., 5., 7.])
        f2 = pt.tensor([11., 13., 17., 19.])
        self.f = pt.stack([f1, f2])
        self.expected = 0.5 * (f1 + f2)

    def test_1d(self):
        actual = self.mix(self.f)
        pt.testing.assert_close(actual, self.expected)

    def test_2d(self):
        actual = self.mix(self.f.tile(3, 1, 1))
        pt.testing.assert_close(actual, self.expected.tile(3, 1))

    def test_3d(self):
        actual = self.mix(self.f.tile(5, 3, 1, 1))
        pt.testing.assert_close(actual, self.expected.tile(5, 3, 1))

    def test_4d(self):
        actual = self.mix(self.f.tile(7, 5, 3, 1, 1))
        pt.testing.assert_close(actual, self.expected.tile(7, 5, 3, 1))

    def test_empty_dims(self):
        actual = self.mix(self.f.tile(5, 0, 1, 1))
        pt.testing.assert_close(actual, self.expected.tile(5, 0, 1))


if __name__ == '__main__':
    unittest.main()
