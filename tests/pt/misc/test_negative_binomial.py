import unittest
from unittest.mock import patch, Mock
import torch as pt
import torch.nn as ptn
from swak.pt.misc import NegativeBinomialFinalizer


class TestNegativeBinomialFinalizer(unittest.TestCase):

    def setUp(self):
        self.default = NegativeBinomialFinalizer(8)
        self.custom = NegativeBinomialFinalizer(
            mod_dim=4,
            bias=False,
            dtype=pt.float64,
            beta=0.75,
            threshold=10
        )
        self.expected_mu = pt.tensor(0.75).exp().log1p() * 4.0 / 3.0
        self.expected_alpha = (
            self.expected_mu * (1.0 + self.expected_mu * self.expected_mu)
        ).sqrt()

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.default, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.default.mod_dim, int)
        self.assertEqual(8, self.default.mod_dim)

    def test_has_bias(self):
        self.assertTrue(hasattr(self.default, 'bias'))

    def test_bias(self):
        self.assertIsInstance(self.default.bias, bool)
        self.assertTrue(self.default.bias)

    def test_custom_bias(self):
        self.assertFalse(self.custom.bias)

    def test_has_device(self):
        self.assertTrue(hasattr(self.default, 'device'))

    def test_device(self):
        self.assertEqual(pt.device('cpu'), self.default.device)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.default, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.default.dtype, pt.float)

    def test_custom_dtype(self):
        self.assertIs(pt.float64, self.custom.dtype)

    def test_has_beta(self):
        self.assertTrue(hasattr(self.default, 'beta'))

    def test_default_beta(self):
        self.assertEqual(1.0, self.default.beta)

    def test_custom_beta(self):
        self.assertEqual(0.75, self.custom.beta)

    def test_has_threshold(self):
        self.assertTrue(hasattr(self.default, 'threshold'))

    def test_default_threshold(self):
        self.assertEqual(20.0, self.default.threshold)

    def test_custom_threshold(self):
        self.assertEqual(10.0, self.custom.threshold)

    def test_has_mu(self):
        self.assertTrue(hasattr(self.default, 'mu'))

    def test_default_mu(self):
        self.assertIsInstance(self.default.mu, ptn.Linear)
        self.assertEqual(8, self.default.mu.in_features)
        self.assertEqual(1, self.default.mu.out_features)
        self.assertIsInstance(self.default.mu.bias, pt.Tensor)

    def test_custom_mu(self):
        self.assertEqual(4, self.custom.mu.in_features)
        self.assertEqual(1, self.custom.mu.out_features)
        self.assertIsNone(self.custom.mu.bias)

    def test_has_alpha(self):
        self.assertTrue(hasattr(self.default, 'alpha'))

    def test_default_alpha(self):
        self.assertIsInstance(self.default.alpha, ptn.Linear)
        self.assertEqual(8, self.default.alpha.in_features)
        self.assertEqual(1, self.default.alpha.out_features)
        self.assertIsInstance(self.default.alpha.bias, pt.Tensor)

    def test_custom_alpha(self):
        self.assertEqual(4, self.custom.alpha.in_features)
        self.assertEqual(1, self.custom.alpha.out_features)
        self.assertIsNone(self.custom.alpha.bias)

    def test_has_activate(self):
        self.assertTrue(hasattr(self.default, 'activate'))

    def test_activate(self):
        self.assertIsInstance(self.default.activate, ptn.Softplus)

    def test_default_activate_params(self):
        self.assertEqual(self.default.beta, self.default.activate.beta)
        self.assertEqual(
            self.default.threshold,
            self.default.activate.threshold
        )

    def test_custom_activate_params(self):
        self.assertEqual(self.custom.beta, self.custom.activate.beta)
        self.assertEqual(
            self.custom.threshold,
            self.custom.activate.threshold
        )

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

    def test_has_new(self):
        self.assertTrue(hasattr(self.default, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.default.new))

    def test_call_new(self):
        new = self.default.new()
        self.assertIsInstance(new, NegativeBinomialFinalizer)
        self.assertIsNot(new, self.default)
        self.assertEqual(self.default.mod_dim, new.mod_dim)
        self.assertEqual(self.default.bias, new.bias)
        self.assertEqual(self.default.beta, new.beta)
        self.assertEqual(self.default.threshold, new.threshold)
        self.assertIsNot(self.default.activate, new.activate)
        self.assertIsNot(self.default.mu, new.mu)
        self.assertIsNot(self.default.alpha, new.alpha)
        self.assertEqual(self.default.device, new.device)
        self.assertIs(self.default.dtype, new.dtype)

    def test_callable(self):
        self.assertTrue(callable(self.custom))

    def test_1d(self):
        self.custom.mu.weight.data = pt.ones(1, 4) / 4.0
        self.custom.alpha.weight.data = pt.ones(1, 4) / 4.0
        inp = pt.ones(4)
        actual_1, actual_2 = self.custom(inp)
        ones = pt.ones(1)
        pt.testing.assert_close(actual_1, self.expected_mu * ones)
        pt.testing.assert_close(actual_2, self.expected_alpha * ones)

    def test_2d(self):
        self.custom.mu.weight.data = pt.ones(1, 4) / 4.0
        self.custom.alpha.weight.data = pt.ones(1, 4) / 4.0
        inp = pt.ones(3, 4)
        actual_1, actual_2 = self.custom(inp)
        ones = pt.ones(3, 1)
        pt.testing.assert_close(actual_1, self.expected_mu * ones)
        pt.testing.assert_close(actual_2, self.expected_alpha * ones)

    def test_3d(self):
        self.custom.mu.weight.data = pt.ones(1, 4) / 4.0
        self.custom.alpha.weight.data = pt.ones(1, 4) / 4.0
        inp = pt.ones(2, 3, 4)
        actual_1, actual_2 = self.custom(inp)
        ones = pt.ones(2, 3, 1)
        pt.testing.assert_close(actual_1, self.expected_mu * ones)
        pt.testing.assert_close(actual_2, self.expected_alpha * ones)

    def test_4d(self):
        self.custom.mu.weight.data = pt.ones(1, 4) / 4.0
        self.custom.alpha.weight.data = pt.ones(1, 4) / 4.0
        inp = pt.ones(5, 2, 3, 4)
        actual_1, actual_2 = self.custom(inp)
        ones = pt.ones(5, 2, 3, 1)
        pt.testing.assert_close(actual_1, self.expected_mu * ones)
        pt.testing.assert_close(actual_2, self.expected_alpha * ones)

    def test_linear_called(self):
        mock_1 = Mock(return_value=pt.ones(1))
        mock_2 = Mock(return_value=pt.ones(1))
        self.default.mu.forward = mock_1
        self.default.alpha.forward = mock_2
        inp = pt.ones(4)
        _ = self.default(inp)
        mock_1.assert_called_once_with(inp)
        mock_2.assert_called_once_with(inp)

    def test_activate_called(self):
        ones = pt.ones(1, 4, device='cpu', dtype=pt.float64) / 4.0
        self.custom.mu.weight.data = ones
        self.custom.alpha.weight.data = ones

        mock = Mock(return_value=pt.ones(1, device='cpu', dtype=pt.float64))
        self.custom.activate.forward = mock

        inp = pt.ones(4, device='cpu', dtype=pt.float64)
        _ = self.custom(inp)

        expected = pt.ones(1, device='cpu', dtype=pt.float64)
        actual_mu = mock.call_args_list[0][0][0]
        actual_alpha = mock.call_args_list[1][0][0]
        pt.testing.assert_close(actual_mu, expected)
        pt.testing.assert_close(actual_alpha, expected)


if __name__ == '__main__':
    unittest.main()
