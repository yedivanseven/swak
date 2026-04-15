import unittest
from unittest.mock import patch, Mock
import torch as pt
import torch.nn as ptn
from swak.pt.misc import Identity, Finalizer


class TestFinalizer(unittest.TestCase):

    def setUp(self):
        self.active1 = ptn.Sigmoid()
        self.active2 = ptn.Softplus()
        self.default = Finalizer(4, self.active1, self.active2)
        self.custom = Finalizer(
            4,
            Identity(),
            Identity(),
            bias=False,
            dtype=pt.float64
        )
        self.custom.finalize[0][0].weight.data = pt.ones(1, 4) / 4.0
        self.custom.finalize[1][0].weight.data = pt.ones(1, 4) / 4.0

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.default, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.default.mod_dim, int)
        self.assertEqual(4, self.default.mod_dim)

    def test_has_device(self):
        self.assertTrue(hasattr(self.default, 'device'))

    def test_device(self):
        self.assertEqual(pt.device('cpu'), self.default.device)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.default, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.default.dtype, pt.float)

    def test_has_activations(self):
        self.assertTrue(hasattr(self.default, 'activations'))

    def test_activations(self):
        self.assertTupleEqual(
            (self.active1, self.active2),
            self.default.activations
        )

    def test_has_n_out(self):
        self.assertTrue(hasattr(self.default, 'n_out'))

    def test_n_out(self):
        self.assertIsInstance(self.default.n_out, int)
        self.assertEqual(2, self.default.n_out)

    def test_has_finalize(self):
        self.assertTrue(hasattr(self.default, 'finalize'))

    def test_finalize(self):
        self.assertIsInstance(self.default.finalize, ptn.ModuleList)
        self.assertEqual(2, len(self.default.finalize))

    @patch('torch.nn.Linear')
    def test_linear_called_once_default(self, mock):
        mock.return_value = self.active1
        _ = Finalizer(4, self.active1)
        mock.assert_called_once_with(
            in_features=4,
            out_features=1,
            bias=True,
            device='cpu',
            dtype=pt.float
        )

    @patch('torch.nn.Linear')
    def test_linear_called_once_custom(self, mock):
        mock.return_value = self.active1
        _ = Finalizer(4, self.active1, bias=False, dtype=pt.float64)
        mock.assert_called_once_with(
            in_features=4,
            out_features=1,
            bias=False,
            device='cpu',
            dtype=pt.float64
        )

    @patch('torch.nn.Linear')
    def test_linear_called_twice(self, mock):
        mock.return_value = self.active1
        _ = Finalizer(4, self.active1, self.active2)
        self.assertEqual(2, mock.call_count)

    @patch('torch.nn.Sigmoid.to')
    def test_to_called_once_default(self, mock):
        mock.return_value = self.active1
        _ = Finalizer(4, self.active1)
        mock.assert_called_once_with(device='cpu', dtype=pt.float)

    @patch('torch.nn.Sigmoid.to')
    def test_to_called_custom(self, mock):
        mock.return_value = self.active1
        _ = Finalizer(4, self.active1, bias=False, dtype=pt.float64)
        mock.assert_called_once_with(device='cpu', dtype=pt.float64)

    @patch('torch.nn.Sigmoid.to')
    def test_to_called_twice(self, mock):
        mock.return_value = self.active1
        _ = Finalizer(4, self.active1, self.active1)
        self.assertEqual(2, mock.call_count)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.default, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.default.reset_parameters))

    def test_call_reset_parameters(self):
        self.default.reset_parameters()

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called(self, mock):
        self.default.reset_parameters()
        self.assertEqual(2, mock.call_count)

    def test_reset_parameters_calls_activations(self):
        finalize = Finalizer(4, ptn.PReLU())
        with patch('torch.nn.PReLU.reset_parameters') as mock:
            finalize.reset_parameters()
            mock.assert_called_once_with()

    @patch('torch.nn.Sigmoid.to', return_value=pt.nn.Sigmoid())
    def test_to_called_on_activate(self, mock):
        self.default.reset_parameters()
        mock.assert_called_once_with(
            device=pt.device('cpu'),
            dtype=pt.float
        )

    def test_has_new(self):
        self.assertTrue(hasattr(self.default, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.default.new))

    def test_call_new(self):
        new = self.default.new()
        self.assertIsInstance(new, Finalizer)
        self.assertIsNot(new, self.default)
        self.assertEqual(self.default.mod_dim, new.mod_dim)
        self.assertTupleEqual(self.default.activations, new.activations)

    def test_callable(self):
        self.assertTrue(callable(self.custom))

    def test_1d(self):
        inp = pt.ones(4)
        actual_1, actual_2 = self.custom(inp)
        expected = pt.ones(1)
        pt.testing.assert_close(actual_1, expected)
        pt.testing.assert_close(actual_2, expected)

    def test_2d(self):
        inp = pt.ones(3, 4)
        actual_1, actual_2 = self.custom(inp)
        expected = pt.ones(3, 1)
        pt.testing.assert_close(actual_1, expected)
        pt.testing.assert_close(actual_2, expected)

    def test_3d(self):
        inp = pt.ones(2, 3, 4)
        actual_1, actual_2 = self.custom(inp)
        expected = pt.ones(2, 3, 1)
        pt.testing.assert_close(actual_1, expected)
        pt.testing.assert_close(actual_2, expected)

    def test_4d(self):
        inp = pt.ones(5, 2, 3, 4)
        actual_1, actual_2 = self.custom(inp)
        expected = pt.ones(5, 2, 3, 1)
        pt.testing.assert_close(actual_1, expected)
        pt.testing.assert_close(actual_2, expected)

    def test_empty_dims(self):
        inp = pt.ones(3, 0, 4)
        actual_1, actual_2 = self.custom(inp)
        expected = pt.ones(3, 0, 1)
        pt.testing.assert_close(actual_1, expected)
        pt.testing.assert_close(actual_2, expected)

    def test_linear_called(self):
        mock_1 = Mock(return_value=pt.ones(1))
        mock_2 = Mock(return_value=pt.ones(1))
        self.default.finalize[0][0].forward = mock_1
        self.default.finalize[1][0].forward = mock_2
        inp = pt.ones(4)
        _ = self.default(inp)
        mock_1.assert_called_once_with(inp)
        mock_2.assert_called_once_with(inp)

    def test_activations_called(self):
        mock_1 = Mock(return_value=pt.ones(1))
        mock_2 = Mock(return_value=pt.ones(1))
        self.custom.activations[0].forward = mock_1
        self.custom.activations[1].forward = mock_2
        inp = pt.ones(4)
        _ = self.custom(inp)
        actual_1 = mock_1.call_args[0][0]
        actual_2 = mock_2.call_args[0][0]
        pt.testing.assert_close(actual_1, pt.ones(1))
        pt.testing.assert_close(actual_2, pt.ones(1))


if __name__ == '__main__':
    unittest.main()
