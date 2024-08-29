import unittest
import torch as pt
from swak.pt.mix import ArgsSumMixer, StackSumMixer


class TestArgsSumMixer(unittest.TestCase):

    def setUp(self):
        self.mix = ArgsSumMixer(2)

    def test_has_n_features(self):
        self.assertTrue(hasattr(self.mix, 'n_features'))

    def test_n_features(self):
        self.assertIsInstance(self.mix.n_features, int)
        self.assertEqual(2, self.mix.n_features)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.mix, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.mix.reset_parameters))

    def test_call_reset_parameters(self):
        self.mix.reset_parameters()

    def test_new(self):
        self.assertTrue(callable(self.mix.new))

    def test_call_new_default(self):
        new = self.mix.new()
        self.assertIsInstance(new, ArgsSumMixer)
        self.assertEqual(self.mix.n_features, new.n_features)

    def test_call_new(self):
        new = self.mix.new(3)
        self.assertIsInstance(new, ArgsSumMixer)
        self.assertEqual(3, new.n_features)

    def test_callable(self):
        self.assertTrue(callable(self.mix))

    def test_1d(self):
        inp = pt.ones(4)
        actual = self.mix(inp, inp)
        expected = pt.ones(4) * 2 / pt.tensor(2).sqrt()
        pt.testing.assert_close(actual, expected)

    def test_2d(self):
        inp = pt.ones(3, 4)
        actual = self.mix(inp, inp)
        expected = pt.ones(3, 4) * 2 / pt.tensor(2).sqrt()
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(2, 3, 4)
        actual = self.mix(inp, inp)
        expected = pt.ones(2, 3, 4) * 2 / pt.tensor(2).sqrt()
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(1, 2, 3, 4)
        actual = self.mix(inp, inp)
        expected = pt.ones(1, 2, 3, 4) * 2 / pt.tensor(2).sqrt()
        pt.testing.assert_close(actual, expected)

    def test_1_feature(self):
        mix = ArgsSumMixer(1)
        inp = pt.ones(2, 3, 4)
        actual = mix(inp)
        expected = pt.ones(2, 3, 4)
        pt.testing.assert_close(actual, expected)

    def test_3_features(self):
        mix = ArgsSumMixer(3)
        inp = pt.ones(2, 3, 4)
        actual = mix(inp, inp, inp)
        expected = pt.ones(2, 3, 4) * 3 / pt.tensor(3).sqrt()
        pt.testing.assert_close(actual, expected)

    def test_4_features(self):
        mix = ArgsSumMixer(4)
        inp = pt.ones(2, 3, 4)
        actual = mix(inp, inp, inp, inp)
        expected = pt.ones(2, 3, 4) * 4 / pt.tensor(4).sqrt()
        pt.testing.assert_close(actual, expected)


class TestStackSumMixer(unittest.TestCase):

    def setUp(self):
        self.mix = StackSumMixer(2)

    def test_has_n_features(self):
        self.assertTrue(hasattr(self.mix, 'n_features'))

    def test_n_features(self):
        self.assertIsInstance(self.mix.n_features, int)
        self.assertEqual(2, self.mix.n_features)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.mix, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.mix.reset_parameters))

    def test_call_reset_parameters(self):
        self.mix.reset_parameters()

    def test_new(self):
        self.assertTrue(callable(self.mix.new))

    def test_call_new_default(self):
        new = self.mix.new()
        self.assertIsInstance(new, StackSumMixer)
        self.assertEqual(self.mix.n_features, new.n_features)

    def test_call_new(self):
        new = self.mix.new(3)
        self.assertIsInstance(new, StackSumMixer)
        self.assertEqual(3, new.n_features)

    def test_callable(self):
        self.assertTrue(callable(self.mix))

    def test_1d(self):
        inp = pt.ones(2, 4)
        actual = self.mix(inp)
        expected = pt.ones(4) * 2 / pt.tensor(2).sqrt()
        pt.testing.assert_close(actual, expected)

    def test_2d(self):
        inp = pt.ones(3, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(3, 4) * 2 / pt.tensor(2).sqrt()
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(1, 3, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(1, 3, 4) * 2 / pt.tensor(2).sqrt()
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(5, 1, 3, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(5, 1, 3, 4) * 2 / pt.tensor(2).sqrt()
        pt.testing.assert_close(actual, expected)

    def test_1_feature(self):
        mix = StackSumMixer(1)
        inp = pt.ones(2, 1, 4)
        actual = mix(inp)
        expected = pt.ones(2, 4)
        pt.testing.assert_close(actual, expected)

    def test_3_features(self):
        mix = StackSumMixer(3)
        inp = pt.ones(2, 3, 4)
        actual = mix(inp)
        expected = pt.ones(2, 4) * 3 / pt.tensor(3).sqrt()
        pt.testing.assert_close(actual, expected)

    def test_4_features(self):
        mix = StackSumMixer(4)
        inp = pt.ones(2, 4, 4)
        actual = mix(inp)
        expected = pt.ones(2, 4) * 4 / pt.tensor(4).sqrt()
        pt.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
