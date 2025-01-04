import unittest
from unittest.mock import patch
import torch as pt
from torch.nn import Linear, Softmax, GELU, Sigmoid, PReLU
from swak.pt.misc import identity
from swak.pt.mix.weighted import GatedSumMixer


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.mix = GatedSumMixer(4, 3)

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

    def test_has_gate(self):
        self.assertTrue(hasattr(self.mix, 'gate'))

    def test_gate(self):
        self.assertIsInstance(self.mix.gate, Sigmoid)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.mix, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.mix.kwargs)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = GatedSumMixer(4, 3)
        mock.assert_called_once_with(12, 6)

    def test_has_coeffs(self):
        self.assertTrue(hasattr(self.mix, 'coeffs'))

    def test_coeffs(self):
        self.assertIsInstance(self.mix.coeffs, Linear)
        self.assertEqual(12, self.mix.coeffs.in_features)
        self.assertEqual(6, self.mix.coeffs.out_features)

    def test_has_norm(self):
        self.assertTrue(hasattr(self.mix, 'norm'))

    def test_norm(self):
        self.assertIsInstance(self.mix.norm, Softmax)

    def test_has_importance(self):
        self.assertTrue(hasattr(self.mix, 'importance'))

    def test_importance(self):
        self.assertTrue(callable(self.mix.importance))

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.mix, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.mix.reset_parameters))

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called_on_instantiation(self, linear):
        gate = PReLU()
        with patch('torch.nn.PReLU.reset_parameters') as mock:
            _ = GatedSumMixer(4, 3, gate)
            self.assertEqual(1, mock.call_count)
            self.assertEqual(1, linear.call_count)

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called(self, mock):
        self.mix.reset_parameters()
        mock.assert_called_once_with()

    def test_reset_parameters_called_on_gate(self):
        mix = GatedSumMixer(4, 3, PReLU())
        with patch('torch.nn.PReLU.reset_parameters') as gate:
            mix.reset_parameters()
            self.assertEqual(1, gate.call_count)

    def test_has_new(self):
        self.assertTrue(hasattr(self.mix, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.mix.new))

    def test_call_new(self):
        new = self.mix.new()
        self.assertIsInstance(new, GatedSumMixer)
        self.assertEqual(self.mix.mod_dim, new.mod_dim)
        self.assertEqual(self.mix.n_features, new.n_features)
        self.assertIs(self.mix.gate, new.gate)
        self.assertDictEqual(self.mix.kwargs, new.kwargs)


class TestAttributes(unittest.TestCase):

    def test_gate(self):
        gate = GELU()
        mix = GatedSumMixer(4, 3, gate)
        self.assertIs(mix.gate, gate)

    def test_kwargs(self):
        mix = GatedSumMixer(4, 3, bias=False)
        self.assertDictEqual({'bias': False}, mix.kwargs)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = GatedSumMixer(4, 2, bias=False)
        mock.assert_called_once_with(8, 4, bias=False)

    def test_new_called(self):
        gate = GELU()
        mix = GatedSumMixer(4, 3)
        new = mix.new(5, 4, gate, bias=False)
        self.assertEqual(5, new.mod_dim)
        self.assertEqual(4, new.n_features)
        self.assertIs(new.gate, gate)
        self.assertDictEqual({'bias': False}, new.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mix = GatedSumMixer(4, 2, bias=False)
        self.mix.coeffs.weight.data = pt.ones(4, 8)

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

    def test_no_features(self):
        mix = GatedSumMixer(4, 0)
        inp = pt.ones(3, 0, 4)
        actual = mix(inp)
        expected = pt.zeros(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_1_feature(self):
        mix = GatedSumMixer(4, 1, bias=False)
        mix.coeffs.weight.data = pt.ones(2, 4)
        inp = pt.ones(3, 1, 4)
        actual = mix(inp)
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_5_features(self):
        mix = GatedSumMixer(4, 5, bias=False)
        mix.coeffs.weight.data = pt.ones(10, 20)
        inp = pt.ones(3, 5, 4)
        actual = mix(inp)
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Linear.forward')
    def test_linear_called(self, mock):
        mock.return_value = pt.ones(3, 4)
        inp = pt.ones(3, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 8)
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Sigmoid.forward')
    def test_gate_called(self, mock):
        mock.return_value = pt.ones(3, 2)
        inp = pt.ones(3, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 2) * 8
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Softmax.forward')
    def test_norm_called(self, mock):
        mix = GatedSumMixer(4, 2, identity, bias=False)
        mix.coeffs.weight.data = pt.ones(4, 8)
        mock.return_value = pt.ones(3, 2) / 2
        inp = pt.ones(3, 2, 4)
        _ = mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 2) * 64
        pt.testing.assert_close(actual, expected)


class TestImportance(unittest.TestCase):

    def setUp(self):
        self.mix = GatedSumMixer(4, 2, bias=False)
        self.mix.coeffs.weight.data = pt.ones(4, 8)

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
        mix = GatedSumMixer(4, 1, bias=False)
        mix.coeffs.weight.data = pt.ones(2, 4)
        inp = pt.ones(3, 1, 4)
        actual = mix.importance(inp)
        expected = pt.ones(3, 1)
        pt.testing.assert_close(actual, expected)

    def test_5_features(self):
        mix = GatedSumMixer(4, 5, bias=False)
        mix.coeffs.weight.data = pt.ones(10, 20)
        inp = pt.ones(3, 5, 4)
        actual = mix.importance(inp)
        expected = pt.ones(3, 5) * 0.2
        pt.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
