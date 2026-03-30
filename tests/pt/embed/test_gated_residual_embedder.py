import unittest
from unittest.mock import patch, Mock
import torch as pt
from torch.nn import Sigmoid, Linear, GELU, ELU, Dropout
from swak.pt.misc import identity
from swak.pt.embed import GatedResidualEmbedder

# ToDo: Test for drop!
class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.embed = GatedResidualEmbedder(4)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.embed, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.embed.mod_dim, int)
        self.assertEqual(4, self.embed.mod_dim)

    def test_has_activate(self):
        self.assertTrue(hasattr(self.embed, 'activate'))

    def test_activate(self):
        self.assertIsInstance(self.embed.activate, ELU)

    def test_has_gate(self):
        self.assertTrue(hasattr(self.embed, 'gate'))

    def test_gate(self):
        self.assertIsInstance(self.embed.gate, Sigmoid)

    def test_has_bias(self):
        self.assertTrue(hasattr(self.embed, 'bias'))

    def test_bias(self):
        self.assertIsInstance(self.embed.bias, bool)
        self.assertTrue(self.embed.bias)

    def test_has_dropout(self):
        self.assertTrue(hasattr(self.embed, 'dropout'))

    def test_dropout(self):
        self.assertIsInstance(self.embed.dropout, float)
        self.assertEqual(self.embed.dropout, 0.0)

    def test_has_drop(self):
        self.assertTrue(hasattr(self.embed, 'drop'))

    def test_drop(self):
        self.assertIsInstance(self.embed.drop, Dropout)

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

    def test_has_widen(self):
        self.assertTrue(hasattr(self.embed, 'widen'))

    def test_widen(self):
        self.assertIsInstance(self.embed.widen, Linear)

    @patch('torch.nn.Linear', return_value=pt.nn.Linear(1, 8))
    def test_linear_called(self, mock):
        _ = GatedResidualEmbedder(4)
        args_1, args_2 = mock.call_args_list
        self.assertTupleEqual((1, 4, True, 'cpu', pt.float), args_1[0])
        self.assertTupleEqual((4, 8, True, 'cpu', pt.float), args_2[0])

    @patch('torch.nn.Sigmoid.to')
    @patch('torch.nn.ELU.to')
    def test_to_called_on_instantiation(self, activate, gate):
        _ = GatedResidualEmbedder(4)
        activate.assert_called_once_with(
            device='cpu',
            dtype=pt.float32
        )
        gate.assert_called_once_with(
            device='cpu',
            dtype=pt.float32
        )

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.embed, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.embed.reset_parameters))

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called(self, mock):
        self.embed.reset_parameters()
        self.assertEqual(2, mock.call_count)

    @patch('torch.nn.ELU.to', return_value=pt.nn.ELU())
    def test_to_called_on_activation(self, mock):
        self.embed.reset_parameters()
        mock.assert_called_once_with(
            device=pt.device('cpu'),
            dtype=pt.float32
        )

    @patch('torch.nn.Sigmoid.to', return_value=pt.nn.Sigmoid())
    def test_to_called_on_gate(self, mock):
        self.embed.reset_parameters()
        mock.assert_called_once_with(
            device=pt.device('cpu'),
            dtype=pt.float32
        )

    def test_has_new(self):
        self.assertTrue(hasattr(self.embed, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.embed.new))

    def test_call_new(self):
        new = self.embed.new()
        self.assertIsInstance(new, GatedResidualEmbedder)
        self.assertEqual(self.embed.mod_dim, new.mod_dim)
        self.assertIs(new.activate, self.embed.activate)
        self.assertIs(new.gate, self.embed.gate)
        self.assertEqual(new.dropout, self.embed.dropout)
        self.assertEqual(self.embed.inp_dim, new.inp_dim)
        self.assertEqual(self.embed.bias, new.bias)
        self.assertEqual(self.embed.dtype, new.dtype)
        self.assertEqual(self.embed.device, new.device)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.embed = GatedResidualEmbedder(
            4,
            GELU(),
            ELU(),
            False,
            0.1,
            2,
            dtype=pt.float64
        )

    def test_activate(self):
        self.assertIsInstance(self.embed.activate, GELU)

    def test_gate(self):
        self.assertIsInstance(self.embed.gate, ELU)

    def test_bias(self):
        self.assertIsInstance(self.embed.bias, bool)
        self.assertFalse(self.embed.bias)

    def test_dropout(self):
        self.assertEqual(self.embed.dropout, 0.1)

    def test_inp_dim(self):
        self.assertIsInstance(self.embed.inp_dim, int)
        self.assertEqual(2, self.embed.inp_dim)

    @patch('torch.nn.Linear', return_value=pt.nn.Linear(1, 8))
    def test_linear_called(self, mock):
        _ = GatedResidualEmbedder(4, inp_dim=2, bias=False, dtype=pt.float64)
        args_1, args_2 = mock.call_args_list
        self.assertTupleEqual((2, 4, False, 'cpu', pt.float64), args_1[0])
        self.assertTupleEqual((4, 8, False, 'cpu', pt.float64), args_2[0])

    def test_reset_parameters_called_on_instantiation(self):
        activation = pt.nn.PReLU()
        gate = pt.nn.PReLU()
        with patch.object(
                activation, 'reset_parameters'
        ) as mock_activation, patch.object(
                gate, 'reset_parameters'
        ) as mock_gate:
            _ = GatedResidualEmbedder(4, activation, gate)
            mock_activation.assert_called_once_with()
            mock_gate.assert_called_once_with()

    def test_reset_parameters_not_called_on_instantiation(self):
        activation = pt.nn.functional.relu
        gate = pt.nn.functional.relu
        _ = GatedResidualEmbedder(4, activation, gate)

    def test_to_called_on_instantiation(self):
        activation = pt.nn.ReLU()
        gate = pt.nn.ReLU()
        with patch.object(
            activation, 'to', return_value=activation
        ) as mock_activation, patch.object(
            gate, 'to', return_value=gate
        ) as mock_gate:
            _ = GatedResidualEmbedder(4, activation, gate, dtype=pt.float64)
            mock_activation.assert_called_once_with(
                device='cpu',
                dtype=pt.float64
            )
            mock_gate.assert_called_once_with(
                device='cpu',
                dtype=pt.float64
            )

    def test_reset_parameters_called_on_activation_gate(self):
        activation = pt.nn.PReLU()
        gate = pt.nn.PReLU()
        embed = GatedResidualEmbedder(4, activation, gate)
        with patch.object(
                activation, 'reset_parameters'
        ) as mock_activation, patch.object(
            gate, 'reset_parameters'
        ) as mock_gate:
            embed.reset_parameters()
            mock_activation.assert_called_once_with()
            mock_gate.assert_called_once_with()

    def test_to_called_on_activation_gate(self):
        activation = pt.nn.ReLU()
        gate = pt.nn.ReLU()
        embed = GatedResidualEmbedder(4, activation, gate, dtype=pt.float64)
        with patch.object(
                activation, 'to', return_value=activation
        ) as mock_activation, patch.object(
            gate, 'to', return_value=gate
        ) as mock_gate:
            embed.reset_parameters()
            mock_activation.assert_called_once_with(
                device=pt.device('cpu'),
                dtype=pt.float64
            )
            mock_gate.assert_called_once_with(
                device=pt.device('cpu'),
                dtype=pt.float64
            )

    def test_dtype(self):
        self.assertIs(self.embed.dtype, pt.float64)
        embed = self.embed.to(pt.float16)
        self.assertIs(embed.dtype, pt.float16)


class TestUsageSingleFeature(unittest.TestCase):

    def setUp(self):
        self.embed = GatedResidualEmbedder(4, identity, identity, bias=False)
        self.embed.embed.weight.data = pt.ones(4, 1)
        self.embed.widen.weight.data = pt.ones(8, 4)

    def test_callable(self):
        self.assertTrue(callable(self.embed))

    def test_1d(self):
        inp = pt.ones(1)
        actual = self.embed(inp)
        expected = pt.ones(4) * 16 + 1
        pt.testing.assert_close(actual, expected)

    def test_2d(self):
        inp = pt.ones(3, 1)
        actual = self.embed(inp)
        expected = pt.ones(3, 4) * 16 + 1
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(2, 3, 1)
        actual = self.embed(inp)
        expected = pt.ones(2, 3, 4) * 16 + 1
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(1, 2, 3, 1)
        actual = self.embed(inp)
        expected = pt.ones(1, 2, 3, 4) * 16 + 1
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(3, 0, 1)
        actual = self.embed(inp)
        expected = pt.ones(3, 0, 4) * 16 + 1
        pt.testing.assert_close(actual, expected)

    def test_embed_called(self):
        mock = Mock(return_value=pt.ones(3, 4))
        self.embed.embed.forward = mock
        inp = pt.ones(3, 1)
        _ = self.embed(inp)
        mock.assert_called_once_with(inp)

    def test_activate_called(self):
        mock = Mock(return_value=pt.ones(3, 4))
        self.embed.activate = mock
        inp = pt.ones(3, 1)
        _ = self.embed(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Dropout.forward', return_value=pt.ones(3, 4))
    def test_drop_called(self, drop):
        embed = Mock(return_value=pt.ones(3, 4))
        self.embed.embed.forward = embed
        inp = pt.ones(3, 1)
        _ = self.embed(inp)
        actual = drop.call_args[0][0]
        expected = pt.ones(3, 4) * 16
        pt.testing.assert_close(actual, expected)

    def test_widen_called(self):
        mock = Mock(return_value=pt.ones(3, 8))
        self.embed.widen.forward = mock
        inp = pt.ones(3, 1)
        _ = self.embed(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_gate_called(self):
        mock = Mock(return_value=pt.ones(3, 4))
        self.embed.gate = mock
        inp = pt.ones(3, 1)
        _ = self.embed(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 4) * 4
        pt.testing.assert_close(actual, expected)


class TestUsageMultiFeature(unittest.TestCase):

    def setUp(self):
        self.embed = GatedResidualEmbedder(
            4,
            identity,
            identity,
            inp_dim=2,
            bias=False
        )
        self.embed.embed.weight.data = pt.ones(4, 2) / 2
        self.embed.widen.weight.data = pt.ones(8, 4) / 4

    def test_1d(self):
        inp = pt.ones(2)
        actual = self.embed(inp)
        expected = pt.ones(4) * 2
        pt.testing.assert_close(actual, expected)

    def test_2d(self):
        inp = pt.ones(3, 2)
        actual = self.embed(inp)
        expected = pt.ones(3, 4) * 2
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(2, 3, 2)
        actual = self.embed(inp)
        expected = pt.ones(2, 3, 4) * 2
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(1, 2, 3, 2)
        actual = self.embed(inp)
        expected = pt.ones(1, 2, 3, 4) * 2
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(3, 0, 2)
        actual = self.embed(inp)
        expected = pt.ones(3, 0, 4)
        pt.testing.assert_close(actual, expected)

    def test_embed_called(self):
        mock = Mock(return_value=pt.ones(3, 4))
        self.embed.embed.forward = mock
        inp = pt.ones(3, 2)
        _ = self.embed(inp)
        mock.assert_called_once_with(inp)

    def test_activate_called(self):
        mock = Mock(return_value=pt.ones(3, 4))
        self.embed.activate = mock
        inp = pt.ones(3, 2)
        _ = self.embed(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Dropout.forward', return_value=pt.ones(3, 4))
    def test_drop_called(self, drop):
        embed = Mock(return_value=pt.ones(3, 4))
        self.embed.embed.forward = embed
        inp = pt.ones(3, 2)
        _ = self.embed(inp)
        actual = drop.call_args[0][0]
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_widen_called(self):
        mock = Mock(return_value=pt.ones(3, 8))
        self.embed.widen.forward = mock
        inp = pt.ones(3, 2)
        _ = self.embed(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_gate_called(self):
        mock = Mock(return_value=pt.ones(3, 4))
        self.embed.gate = mock
        inp = pt.ones(3, 2)
        _ = self.embed(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
