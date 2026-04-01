import unittest
from unittest.mock import patch, Mock
import torch as pt
import torch.nn as ptn
from swak.pt.blocks import GatedActivatedBlock
from swak.pt.misc import identity


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.block = GatedActivatedBlock(4)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.block, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.block.mod_dim, int)
        self.assertEqual(4, self.block.mod_dim)

    def test_has_activate(self):
        self.assertTrue(hasattr(self.block, 'activate'))

    def test_activate(self):
        self.assertIsInstance(self.block.activate, ptn.ELU)

    def test_has_gate(self):
        self.assertTrue(hasattr(self.block, 'gate'))

    def test_gate(self):
        self.assertIsInstance(self.block.gate, ptn.Sigmoid)

    def test_has_bias(self):
        self.assertTrue(hasattr(self.block, 'bias'))

    def test_bias(self):
        self.assertIsInstance(self.block.bias, bool)
        self.assertTrue(self.block.bias)

    def test_has_device(self):
        self.assertTrue(hasattr(self.block, 'device'))

    def test_device(self):
        self.assertEqual(pt.device('cpu'), self.block.device)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.block, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.block.dtype, pt.float)

    def test_has_project(self):
        self.assertTrue(hasattr(self.block, 'project'))

    def test_project(self):
        self.assertIsInstance(self.block.project, ptn.Linear)
        self.assertEqual(4, self.block.project.in_features)
        self.assertEqual(4, self.block.project.out_features)
        self.assertIsInstance(self.block.project.bias, pt.Tensor)

    def test_has_rotate(self):
        self.assertTrue(hasattr(self.block, 'rotate'))

    def test_rotate(self):
        self.assertIsInstance(self.block.rotate, ptn.Linear)
        self.assertEqual(4, self.block.rotate.in_features)
        self.assertEqual(4, self.block.rotate.out_features)
        self.assertIsInstance(self.block.rotate.bias, pt.Tensor)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.block, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.block.reset_parameters))

    def test_reset_parameters_called_on_instantiation(self):
        activate = ptn.PReLU()
        gate = ptn.PReLU()
        with patch('torch.nn.PReLU.reset_parameters') as mock:
            _ = GatedActivatedBlock(4, activate, gate)
            self.assertEqual(2, mock.call_count)

    @patch('torch.nn.Sigmoid.to')
    @patch('torch.nn.ELU.to')
    def test_to_called_on_instantiation(self, activate, gate):
        _ = GatedActivatedBlock(4)
        activate.assert_called_once_with(
            device='cpu',
            dtype=pt.float32
        )
        gate.assert_called_once_with(
            device='cpu',
            dtype=pt.float32
        )

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called(self, mock):
        self.block.reset_parameters()
        self.assertEqual(2, mock.call_count)

    def test_reset_parameters_called_on_activation(self):
        block = GatedActivatedBlock(4, ptn.PReLU(), ptn.PReLU())
        with patch('torch.nn.PReLU.reset_parameters') as activation:
            block.reset_parameters()
            self.assertEqual(2, activation.call_count)

    @patch('torch.nn.ELU.to', return_value=pt.nn.ELU())
    def test_to_called_on_activation(self, mock):
        self.block.reset_parameters()
        mock.assert_called_once_with(
            device=pt.device('cpu'),
            dtype=pt.float32
        )

    @patch('torch.nn.Sigmoid.to', return_value=pt.nn.Sigmoid())
    def test_to_called_on_gate(self, mock):
        self.block.reset_parameters()
        mock.assert_called_once_with(
            device=pt.device('cpu'),
            dtype=pt.float32
        )

    def test_has_new(self):
        self.assertTrue(hasattr(self.block, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.block.new))

    def test_call_new(self):
        new = self.block.new()
        self.assertIsInstance(new, GatedActivatedBlock)
        self.assertEqual(self.block.mod_dim, new.mod_dim)
        self.assertEqual(self.block.bias, new.bias)
        self.assertIs(self.block.gate, new.gate)
        self.assertEqual(self.block.device, new.device)
        self.assertIs(self.block.activate, new.activate)
        self.assertIs(self.block.gate, new.gate)
        self.assertIsNot(self.block.rotate, new.rotate)
        self.assertIsNot(self.block.project, new.project)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.block = GatedActivatedBlock(
            8,
            ptn.ReLU(),
            ptn.GELU(),
            bias=False,
            dtype=pt.float64
        )

    def test_mod_dim(self):
        self.assertIsInstance(self.block.mod_dim, int)
        self.assertEqual(8, self.block.mod_dim)

    def test_bias(self):
        self.assertFalse(self.block.bias)

    def test_activate(self):
        self.assertIsInstance(self.block.activate, ptn.ReLU)

    def test_gate(self):
        self.assertIsInstance(self.block.gate, ptn.GELU)

    def test_project(self):
        self.assertIsInstance(self.block.project, ptn.Linear)
        self.assertEqual(8, self.block.project.in_features)
        self.assertEqual(8, self.block.project.out_features)
        self.assertIsNone(self.block.project.bias)

    def test_rotate(self):
        self.assertIsNone(self.block.rotate.bias)
        self.assertEqual(8, self.block.rotate.in_features)
        self.assertEqual(8, self.block.rotate.out_features)
        self.assertIsNone(self.block.rotate.bias)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.block = GatedActivatedBlock(4, identity, identity, bias=False)
        self.block.project.weight.data = pt.ones(4, 4)
        self.block.rotate.weight.data = pt.ones(4, 4)

    def test_callable(self):
        self.assertTrue(callable(self.block))

    def test_1d(self):
        inp = pt.ones(4)
        actual = self.block(inp)
        expected = pt.ones(4) * 64
        pt.testing.assert_close(actual, expected)

    def test_2d(self):
        inp = pt.ones(3, 4)
        actual = self.block(inp)
        expected = pt.ones(3, 4) * 64
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(2, 3, 4)
        actual = self.block(inp)
        expected = pt.ones(2, 3, 4) * 64
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(5, 2, 3, 4)
        actual = self.block(inp)
        expected = pt.ones(5, 2, 3, 4) * 64
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(3, 0, 4)
        actual = self.block(inp)
        expected = pt.ones(3, 0, 4) * 64
        pt.testing.assert_close(actual, expected)

    def test_project_called(self):
        with patch.object(
                self.block.project,
                'forward',
                return_value=pt.ones(4)
        ) as mock:
            expected = pt.ones(4)
            _ = self.block(expected)
            mock.assert_called_once_with(expected)

    def test_activate_called(self):
        mock = Mock(return_value=pt.ones(4))
        self.block.activate = mock
        inp = pt.ones(4)
        _ = self.block(inp)
        mock.assert_called_once()
        actual = mock.call_args[0][0]
        expected = pt.ones(4) * 4
        pt.testing.assert_close(actual, expected)

    def test_rotate_called(self):
        with patch.object(
                self.block.rotate,
                'forward',
                return_value=pt.ones(4)
        ) as mock:
            inp = pt.ones(4)
            _ = self.block(inp)
            actual = mock.call_args[0][0]
            expected = pt.ones(4) * 4
            pt.testing.assert_close(actual, expected)

    def test_gate_called(self):
        mock = Mock(return_value=pt.ones(4))
        self.block.gate = mock
        inp = pt.ones(4)
        _ = self.block(inp)
        mock.assert_called_once()
        actual = mock.call_args[0][0]
        expected = pt.ones(4) * 16
        pt.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
