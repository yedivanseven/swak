import unittest
from unittest.mock import patch, Mock
import torch as pt
import torch.nn as ptn
from swak.pt.blocks import GatedBlock
from swak.pt.misc import identity


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.block = GatedBlock(4)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.block, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.block.mod_dim, int)
        self.assertEqual(4, self.block.mod_dim)

    def test_has_gate(self):
        self.assertTrue(hasattr(self.block, 'gate'))

    def test_gate(self):
        self.assertIsInstance(self.block.gate, ptn.Sigmoid)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.block, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.block.kwargs)

    def test_has_widen(self):
        self.assertTrue(hasattr(self.block, 'widen'))

    def test_widen(self):
        self.assertIsInstance(self.block.widen, ptn.Linear)
        self.assertEqual(4, self.block.widen.in_features)
        self.assertEqual(8, self.block.widen.out_features)
        self.assertIsInstance(self.block.widen.bias, pt.Tensor)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.block, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.block.reset_parameters))

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called_on_instantiation(self, linear):
        gate = ptn.PReLU()
        with patch('torch.nn.PReLU.reset_parameters') as mock:
            _ = GatedBlock(4, gate)
            self.assertEqual(1, mock.call_count)
            self.assertEqual(1, linear.call_count)

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called(self, mock):
        self.block.reset_parameters()
        mock.assert_called_once_with()

    def test_reset_parameters_called_on_gate(self):
        block = GatedBlock(4, ptn.PReLU())
        with patch('torch.nn.PReLU.reset_parameters') as gate:
            block.reset_parameters()
            self.assertEqual(1, gate.call_count)

    def test_has_new(self):
        self.assertTrue(hasattr(self.block, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.block.new))

    def test_call_new(self):
        new = self.block.new()
        self.assertIsInstance(new, GatedBlock)
        self.assertEqual(self.block.mod_dim, new.mod_dim)
        self.assertIs(self.block.gate, new.gate)
        self.assertDictEqual(self.block.kwargs, new.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.block = GatedBlock(2.8, ptn.ReLU(), bias=False)

    def test_mod_dim(self):
        self.assertIsInstance(self.block.mod_dim, int)
        self.assertEqual(3, self.block.mod_dim)

    def test_gate(self):
        self.assertIsInstance(self.block.gate, ptn.ReLU)

    def test_kwargs(self):
        self.assertDictEqual({'bias': False}, self.block.kwargs)

    def test_widen(self):
        self.assertIsNone(self.block.widen.bias)
        self.assertEqual(3, self.block.widen.in_features)
        self.assertEqual(6, self.block.widen.out_features)



class TestUsage(unittest.TestCase):

    def setUp(self):
        self.block = GatedBlock(4, identity, bias=False)
        self.block.widen.weight.data = pt.ones(8, 4)

    def test_callable(self):
        self.assertTrue(callable(self.block))

    def test_1d(self):
        inp = pt.ones(4)
        actual = self.block(inp)
        expected = pt.ones(4) * 16
        pt.testing.assert_close(actual, expected)

    def test_2d(self):
        inp = pt.ones(3, 4)
        actual = self.block(inp)
        expected = pt.ones(3, 4) * 16
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(2, 3, 4)
        actual = self.block(inp)
        expected = pt.ones(2, 3, 4) * 16
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(5, 2, 3, 4)
        actual = self.block(inp)
        expected = pt.ones(5, 2, 3, 4) * 16
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(3, 0, 4)
        actual = self.block(inp)
        expected = pt.ones(3, 0, 4) * 16
        pt.testing.assert_close(actual, expected)

    def test_widen_called(self):
        mock = Mock(return_value=pt.ones(8))
        self.block.widen.forward = mock
        expected = pt.ones(4)
        _ = self.block(expected)
        mock.assert_called_once_with(expected)

    def test_gate_called(self):
        mock = Mock(return_value=pt.ones(4))
        self.block.gate = mock
        inp = pt.ones(4)
        _ = self.block(inp)
        mock.assert_called_once()
        actual = mock.call_args[0][0]
        expected = pt.ones(4) * 4
        pt.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
