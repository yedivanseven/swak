import unittest
from unittest.mock import patch, Mock
import torch as pt
from torch.nn import Sigmoid, Linear, GELU, PReLU
from swak.pt.misc import identity
from swak.pt.embed import GatedEmbedder


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.embed = GatedEmbedder(4)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.embed, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.embed.mod_dim, int)
        self.assertEqual(4, self.embed.mod_dim)

    def test_has_gate(self):
        self.assertTrue(hasattr(self.embed, 'gate'))

    def test_gate(self):
        self.assertIsInstance(self.embed.gate, Sigmoid)

    def test_has_bias(self):
        self.assertTrue(hasattr(self.embed, 'bias'))

    def test_bias(self):
        self.assertIsInstance(self.embed.bias, bool)
        self.assertTrue(self.embed.bias)

    def test_has_inp_dim(self):
        self.assertTrue(hasattr(self.embed, 'inp_dim'))

    def test_inp_dim(self):
        self.assertIsInstance(self.embed.inp_dim, int)
        self.assertEqual(1, self.embed.inp_dim)

    def test_has_device(self):
        self.assertTrue(hasattr(self.embed, 'device'))

    def test_device(self):
        self.assertEqual(pt.device('cpu'), self.embed.device)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.embed, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.embed.dtype, pt.float)

    def test_has_embed(self):
        self.assertTrue(hasattr(self.embed, 'embed'))

    def test_embed(self):
        self.assertIsInstance(self.embed.embed, Linear)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = GatedEmbedder(4)
        mock.assert_called_once_with(1, 8, True, 'cpu', pt.float)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.embed, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.embed.reset_parameters))

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called_on_instantiation(self, linear):
        gate = PReLU()
        with patch('torch.nn.PReLU.reset_parameters') as mock:
            _ = GatedEmbedder(4, gate)
            self.assertEqual(1, mock.call_count)
            self.assertEqual(1, linear.call_count)

    def test_to_called_on_instantiation(self):
        gate = PReLU()
        with patch('torch.nn.PReLU.to') as mock:
            _ = GatedEmbedder(4, gate, device='cpu', dtype=pt.float64)
            mock.assert_called_once_with(device='cpu', dtype=pt.float64)

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called(self, mock):
        self.embed.reset_parameters()
        mock.assert_called_once_with()

    def test_reset_parameters_called_on_gate(self):
        embed = GatedEmbedder(4, PReLU())
        with patch('torch.nn.PReLU.reset_parameters') as gate:
            embed.reset_parameters()
            self.assertEqual(1, gate.call_count)

    def test_has_new(self):
        self.assertTrue(hasattr(self.embed, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.embed.new))

    def test_call_new(self):
        new = self.embed.new()
        self.assertIsInstance(new, GatedEmbedder)
        self.assertEqual(self.embed.mod_dim, new.mod_dim)
        self.assertIsInstance(self.embed.gate, Sigmoid)
        self.assertEqual(self.embed.inp_dim, new.inp_dim)
        self.assertEqual(self.embed.bias, new.bias)
        self.assertEqual(self.embed.dtype, new.dtype)
        self.assertEqual(self.embed.device, new.device)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.embed = GatedEmbedder(4, GELU(), False, 2)

    def test_gate(self):
        self.assertIsInstance(self.embed.gate, GELU)

    def test_inp_dim(self):
        self.assertIsInstance(self.embed.inp_dim, int)
        self.assertEqual(2, self.embed.inp_dim)

    def test_bias(self):
        self.assertIsInstance(self.embed.bias, bool)
        self.assertFalse(self.embed.bias)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = GatedEmbedder(4, GELU(), bias=False, dtype=pt.float64)
        mock.assert_called_once_with(1, 8, False, 'cpu', pt.float64)


class TestUsageSingleFeature(unittest.TestCase):

    def setUp(self):
        self.embed = GatedEmbedder(4, identity, False)
        self.embed.embed.weight.data = pt.ones(8, 1)

    def test_callable(self):
        self.assertTrue(callable(self.embed))

    def test_1d(self):
        inp = pt.ones(1)
        actual = self.embed(inp)
        expected = pt.ones(4)
        pt.testing.assert_close(actual, expected)

    def test_2d(self):
        inp = pt.ones(3, 1)
        actual = self.embed(inp)
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(2, 3, 1)
        actual = self.embed(inp)
        expected = pt.ones(2, 3, 4)
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(1, 2, 3, 1)
        actual = self.embed(inp)
        expected = pt.ones(1, 2, 3, 4)
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(3, 0, 1)
        actual = self.embed(inp)
        expected = pt.ones(3, 0, 4)
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Linear.forward', return_value=pt.ones(8))
    def test_linear_called(self, linear):
        inp = pt.ones(1)
        _ = self.embed(inp)
        linear.assert_called_once_with(inp)

    def test_gate_called(self):
        mock = Mock(return_value=pt.ones(4))
        self.embed.gate = mock
        inp = pt.ones(1)
        _ = self.embed(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(4)
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Sigmoid.forward')
    @patch('torch.nn.Linear.forward')
    def test_return_value(self, linear, sigmoid):
        inp = pt.ones(1)
        linear.return_value = pt.tensor([[2.0, 3.0, 1.0, 1.0]])
        sigmoid.return_value = pt.tensor([[0.4, 0.5]])
        embed = GatedEmbedder(2)
        actual = embed(inp)
        pt.testing.assert_close(actual, pt.tensor([[0.8, 1.5]]))


class TestUsageMultiFeature(unittest.TestCase):

    def setUp(self):
        self.embed = GatedEmbedder(4, identity, False, 2)
        self.embed.embed.weight.data = pt.ones(8, 2)

    def test_1d(self):
        inp = pt.ones(2)
        actual = self.embed(inp)
        expected = pt.ones(4) * 4
        pt.testing.assert_close(actual, expected)

    def test_2d(self):
        inp = pt.ones(3, 2)
        actual = self.embed(inp)
        expected = pt.ones(3, 4) * 4
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(2, 3, 2)
        actual = self.embed(inp)
        expected = pt.ones(2, 3, 4) * 4
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(1, 2, 3, 2)
        actual = self.embed(inp)
        expected = pt.ones(1, 2, 3, 4) * 4
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(3, 0, 2)
        actual = self.embed(inp)
        expected = pt.ones(3, 0, 4) * 4
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Linear.forward', return_value=pt.ones(8))
    def test_linear_called(self, linear):
        inp = pt.ones(2)
        _ = self.embed(inp)
        linear.assert_called_once_with(inp)

    def test_gate_called(self):
        mock = Mock(return_value=pt.ones(4))
        self.embed.gate = mock
        inp = pt.ones(2)
        _ = self.embed(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(4) * 2
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Sigmoid.forward')
    @patch('torch.nn.Linear.forward')
    def test_return_value(self, linear, sigmoid):
        inp = pt.ones(1)
        linear.return_value = pt.tensor([[2.0, 3.0, 1.0, 1.0]])
        sigmoid.return_value = pt.tensor([[0.4, 0.5]])
        embed = GatedEmbedder(2, Sigmoid(), inp_dim=2)
        actual = embed(inp)
        pt.testing.assert_close(actual, pt.tensor([[0.8, 1.5]]))


if __name__ == '__main__':
    unittest.main()
