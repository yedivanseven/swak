import unittest
import torch as pt
from swak.pt.mix.weighted import ConstantSumMixer


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.mix = ConstantSumMixer(3)

    def test_has_n_features(self):
        self.assertTrue(hasattr(self.mix, 'n_features'))

    def test_n_features(self):
        self.assertIsInstance(self.mix.n_features, int)
        self.assertEqual(3, self.mix.n_features)

    def test_has_coeffs(self):
        self.assertTrue(hasattr(self.mix, 'coeffs'))

    def test_coeffs(self):
        expected = pt.tensor(1.0 / 3)
        pt.testing.assert_close(self.mix.coeffs, expected)

    def test_has_importance(self):
        self.assertTrue(hasattr(self.mix, 'importance'))

    def test_importance(self):
        self.assertTrue(callable(self.mix.importance))

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.mix, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.mix.reset_parameters))

    def test_reset_parameters_called(self):
        self.mix.reset_parameters()

    def test_has_new(self):
        self.assertTrue(hasattr(self.mix, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.mix.new))

    def test_call_new_defaults(self):
        new = self.mix.new()
        self.assertIsInstance(new, ConstantSumMixer)
        self.assertEqual(self.mix.n_features, new.n_features)

    def test_call_new_update(self):
        new = self.mix.new(2)
        self.assertIsInstance(new, ConstantSumMixer)
        self.assertEqual(2, new.n_features)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mix = ConstantSumMixer(2)

    def test_callable(self):
        self.assertTrue(callable(self.mix))

    def test_2d(self):
        inp = pt.ones(2, 4)
        actual = self.mix(inp)
        expected = pt.ones(4)
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(3, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(5, 3, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(5, 3, 4)
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(5, 0, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(5, 0, 4)
        pt.testing.assert_close(actual, expected)

    def test_1_feature(self):
        mix = ConstantSumMixer(1)
        inp = pt.ones(3, 1, 4)
        actual = mix(inp)
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_5_features(self):
        mix = ConstantSumMixer(5)
        inp = pt.ones(3, 5, 4)
        actual = mix(inp)
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)


class TestImportance(unittest.TestCase):

    def setUp(self):
        self.mix = ConstantSumMixer(2)

    def test_2d(self):
        inp = pt.ones(2, 4)
        actual = self.mix.importance(inp)
        expected = pt.ones(2) * 0.5
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(3, 2, 4)
        actual = self.mix.importance(inp)
        expected = pt.ones(3, 2) * 0.5
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(5, 3, 2, 4)
        actual = self.mix.importance(inp)
        expected = pt.ones(5, 3, 2) * 0.5
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(5, 0, 2, 4)
        actual = self.mix.importance(inp)
        expected = pt.ones(5, 0, 2) * 0.5
        pt.testing.assert_close(actual, expected)

    def test_1_feature(self):
        mix = ConstantSumMixer(1)
        inp = pt.ones(3, 1, 4)
        actual = mix.importance(inp)
        expected = pt.ones(3, 1)
        pt.testing.assert_close(actual, expected)

    def test_5_features(self):
        mix = ConstantSumMixer(5)
        inp = pt.ones(3, 5, 4)
        actual = mix.importance(inp)
        expected = pt.ones(3, 5) * 0.2
        pt.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
