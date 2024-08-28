import unittest
from unittest.mock import patch, Mock
import torch as pt
from torch.nn import Linear
from swak.pt.misc import identity
from swak.pt.mix import ActivatedStackConcatMixer


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.mix = ActivatedStackConcatMixer(4, 3)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.mix, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.mix.mod_dim, int)
        self.assertEqual(4, self.mix.mod_dim)

    def test_has_n_features(self):
        self.assertTrue(hasattr(self.mix, 'n_features'))

    def test_n_features(self):
        self.assertIsInstance(self.mix.n_features, int)
        self.assertEqual(3, self.mix.n_features)

    def test_has_activate(self):
        self.assertTrue(hasattr(self.mix, 'activate'))

    def test_activate(self):
        self.assertIs(self.mix.activate, identity)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.mix, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.mix.kwargs)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = ActivatedStackConcatMixer(4, 3)
        mock.assert_called_once_with(12, 4)

    def test_has_mix(self):
        self.assertTrue(hasattr(self.mix, 'mix'))

    def test_mix(self):
        self.assertIsInstance(self.mix.mix, Linear)
        self.assertEqual(12, self.mix.mix.in_features)
        self.assertEqual(4, self.mix.mix.out_features)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.mix, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.mix.reset_parameters))

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called(self, mock):
        self.mix.reset_parameters()
        mock.assert_called_once_with()

    def test_has_new(self):
        self.assertTrue(hasattr(self.mix, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.mix.new))

    def test_call_new(self):
        new = self.mix.new()
        self.assertIsInstance(new, ActivatedStackConcatMixer)
        self.assertEqual(self.mix.mod_dim, new.mod_dim)
        self.assertEqual(self.mix.n_features, new.n_features)
        self.assertIs(self.mix.activate, new.activate)


class TestAttributes(unittest.TestCase):

    def test_dropout(self):
        activate = pt.nn.SELU()
        mix = ActivatedStackConcatMixer(4, 3, activate)
        self.assertIs(mix.activate, activate)

    def test_kwargs(self):
        mix = ActivatedStackConcatMixer(4, 3, bias=False)
        self.assertDictEqual({'bias': False}, mix.kwargs)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = ActivatedStackConcatMixer(4, 3, bias=False)
        mock.assert_called_once_with(12, 4, bias=False)

    def test_new_called(self):
        activate = pt.nn.SELU()
        mix = ActivatedStackConcatMixer(4, 3)
        new = mix.new(5, 4, activate, bias=False)
        self.assertEqual(5, new.mod_dim)
        self.assertEqual(4, new.n_features)
        self.assertIs(new.activate, activate)
        self.assertDictEqual({'bias': False}, new.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mix = ActivatedStackConcatMixer(4, 2, bias=False)
        self.mix.mix.weight.data = pt.ones(4, 8)

    def test_1d(self):
        inp = pt.ones(2, 4)
        actual = self.mix(inp)
        expected = pt.ones(4) * 8
        pt.testing.assert_close(actual, expected)

    def test_2d(self):
        inp = pt.ones(5, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(5, 4) * 8
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(3, 5, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(3, 5, 4) * 8
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(1, 3, 5, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(1, 3, 5, 4) * 8
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(3, 0, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(3, 0, 4) * 8
        pt.testing.assert_close(actual, expected)

    def test_no_features(self):
        mix = ActivatedStackConcatMixer(4, 0, bias=False)
        inp = pt.ones(3, 0, 4)
        actual = mix(inp)
        expected = pt.zeros(3, 4)
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Linear.forward')
    def test_linear_called(self, mock):
        inp = pt.ones(3, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 8)
        pt.testing.assert_close(actual, expected)

    def test_activate_called(self):
        mock = Mock()
        self.mix.activate = mock
        inp = pt.ones(3, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 4) * 8
        pt.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
