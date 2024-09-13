import unittest
from unittest.mock import patch, Mock
import torch as pt
import torch.nn as ptn

from swak.pt.misc import (
    identity,
    Identity,
    Finalizer,
    NegativeBinomialFinalizer
)


class TestIdentityFunction(unittest.TestCase):

    def test_callable(self):
        self.assertTrue(callable(Identity()))

    def test_arg(self):
        obj = object()
        result = identity(obj)
        self.assertIs(result, obj)

    def test_arg_and_kwargs(self):
        obj = object()
        result = identity(obj, foo='bar', answer=42)
        self.assertIs(result, obj)


class TestIdentityModule(unittest.TestCase):

    def test_empty_instantiation(self):
        _ = Identity()

    def test_instantiation_accepts_args(self):
        _ = Identity(1, 'foo', 4.2)

    def test_instantiation_accepts_kwargs(self):
        _ = Identity(foo=1, bar='baz', answer=4.2)

    def test_instantiation_accepts_kargs_and_kwargs(self):
        _ = Identity(1, 42, foo='bar')

    def test_callable(self):
        self.assertTrue(callable(Identity()))

    def test_arg(self):
        obj = object()
        result = Identity()(obj)
        self.assertIs(result, obj)

    def test_arg_and_kwargs(self):
        obj = object()
        result = Identity()(obj, foo='bar', answer=42)
        self.assertIs(result, obj)

    def test_has_reset_parameters(self):
        i = Identity()
        self.assertTrue(hasattr(i, 'reset_parameters'))

    def test_reset_parameters(self):
        i = Identity()
        self.assertTrue(callable(i.reset_parameters))

    def test_call_reset_parameters(self):
        i = Identity()
        i.reset_parameters()

    def test_has_new(self):
        i = Identity()
        self.assertTrue(hasattr(i, 'new'))

    def test_new(self):
        i = Identity()
        self.assertTrue(callable(i.new))

    def test_call_new(self):
        old = Identity()
        new = old.new()
        self.assertIsInstance(new, Identity)


class TestFinalizer(unittest.TestCase):

    def setUp(self):
        self.active1 = ptn.Sigmoid()
        self.active2 = ptn.Softplus()
        self.empty = Finalizer(4)
        self.default = Finalizer(4, self.active1, self.active2)
        self.custom = Finalizer(4, Identity(), Identity(), bias=False)
        self.custom.finalize[0][0].weight.data = pt.ones(1, 4) / 4.0
        self.custom.finalize[1][0].weight.data = pt.ones(1, 4) / 4.0

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.default, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.default.mod_dim, int)
        self.assertEqual(4, self.default.mod_dim)

    def test_has_activations(self):
        self.assertTrue(hasattr(self.default, 'activations'))

    def test_activations(self):
        self.assertTupleEqual(
            (self.active1, self.active2),
            self.default.activations
        )

    def test_empty_activations(self):
        self.assertTupleEqual((), self.empty.activations)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.default, 'kwargs'))

    def test_default_kwargs(self):
        self.assertDictEqual({}, self.default.kwargs)

    def test_custom_kwargs(self):
        self.assertDictEqual({'bias': False}, self.custom.kwargs)

    def test_has_n_out(self):
        self.assertTrue(hasattr(self.default, 'n_out'))

    def test_n_out(self):
        self.assertIsInstance(self.default.n_out, int)
        self.assertEqual(2, self.default.n_out)

    def test_empty_n_out(self):
        self.assertEqual(0, self.empty.n_out)

    def test_has_finalize(self):
        self.assertTrue(hasattr(self.default, 'finalize'))

    def test_finalize(self):
        self.assertIsInstance(self.default.finalize, ptn.ModuleList)

    @patch('torch.nn.Linear')
    def test_empty_linear_not_called(self, mock):
        _ = Finalizer(4)
        mock.assert_not_called()

    @patch('torch.nn.Linear')
    def test_linear_called_once(self, mock):
        mock.return_value = self.active2
        _ = Finalizer(4, self.active1, bias=False)
        mock.assert_called_once_with(4, 1, bias=False)

    @patch('torch.nn.Linear')
    def test_linear_called_twice(self, mock):
        mock.return_value = self.active2
        _ = Finalizer(4, self.active1, self.active2)
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

    def test_has_new(self):
        self.assertTrue(hasattr(self.default, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.default.new))

    def test_call_new_defaults(self):
        new = self.default.new()
        self.assertIsInstance(new, Finalizer)
        self.assertEqual(self.default.mod_dim, new.mod_dim)
        self.assertTupleEqual(self.default.activations, new.activations)

    def test_call_new_update(self):
        new = self.default.new(8, self.active2, self.active1, bias=False)
        self.assertEqual(8, new.mod_dim)
        self.assertTupleEqual(
            (self.active2, self.active1),
            new.activations
        )
        self.assertDictEqual({'bias': False}, new.kwargs)

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


class TestNegativeBinomialFinalizer(unittest.TestCase):

    def setUp(self):
        self.default = NegativeBinomialFinalizer(8)
        self.custom = NegativeBinomialFinalizer(4, 0.75, 10, bias=False)
        self.custom.mu.weight.data = pt.ones(1, 4) / 4.0
        self.custom.alpha.weight.data = pt.ones(1, 4) / 4.0
        self.expected_mu = pt.tensor(0.75).exp().log1p() * 4.0 / 3.0
        self.expected_alpha = (
            self.expected_mu * (1.0 + self.expected_mu * self.expected_mu)
        ).sqrt()

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.default, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.default.mod_dim, int)
        self.assertEqual(8, self.default.mod_dim)

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

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.default, 'kwargs'))

    def test_default_kwargs(self):
        self.assertDictEqual({}, self.default.kwargs)

    def test_custom_kwargs(self):
        self.assertDictEqual({'bias': False}, self.custom.kwargs)

    def test_has_mu(self):
        self.assertTrue(hasattr(self.default, 'mu'))

    def test_default_mu(self):
        self.assertIsInstance(self.default.mu, ptn.Linear)
        self.assertEqual(8, self.default.mu.in_features)
        self.assertEqual(1, self.default.mu.out_features)
        self.assertTrue(self.default.mu.bias)

    def test_custom_mu(self):
        self.assertEqual(4, self.custom.mu.in_features)
        self.assertEqual(1, self.custom.mu.out_features)
        self.assertFalse(self.custom.mu.bias)

    def test_has_alpha(self):
        self.assertTrue(hasattr(self.default, 'alpha'))

    def test_default_alpha(self):
        self.assertIsInstance(self.default.alpha, ptn.Linear)
        self.assertEqual(8, self.default.alpha.in_features)
        self.assertEqual(1, self.default.alpha.out_features)
        self.assertTrue(self.default.alpha.bias)

    def test_custom_alpha(self):
        self.assertEqual(4, self.custom.alpha.in_features)
        self.assertEqual(1, self.custom.alpha.out_features)
        self.assertFalse(self.custom.alpha.bias)

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

    def test_call_new_defaults(self):
        new = self.default.new()
        self.assertIsInstance(new, NegativeBinomialFinalizer)
        self.assertEqual(self.default.mod_dim, new.mod_dim)
        self.assertEqual(self.default.beta, new.beta)
        self.assertEqual(self.default.threshold, new.threshold)
        self.assertDictEqual(self.default.kwargs, new.kwargs)

    def test_call_new_custom(self):
        new = self.default.new(4, 0.75, 10., bias=False)
        self.assertEqual(4, new.mod_dim)
        self.assertEqual(0.75, new.beta)
        self.assertEqual(10., new.threshold)
        self.assertDictEqual({'bias': False}, new.kwargs)

    def test_callable(self):
        self.assertTrue(callable(self.custom))

    def test_1d(self):
        inp = pt.ones(4)
        actual_1, actual_2 = self.custom(inp)
        ones = pt.ones(1)
        pt.testing.assert_close(actual_1, self.expected_mu * ones)
        pt.testing.assert_close(actual_2, self.expected_alpha * ones)

    def test_2d(self):
        inp = pt.ones(3, 4)
        actual_1, actual_2 = self.custom(inp)
        ones = pt.ones(3, 1)
        pt.testing.assert_close(actual_1, self.expected_mu * ones)
        pt.testing.assert_close(actual_2, self.expected_alpha * ones)

    def test_3d(self):
        inp = pt.ones(2, 3, 4)
        actual_1, actual_2 = self.custom(inp)
        ones = pt.ones(2, 3, 1)
        pt.testing.assert_close(actual_1, self.expected_mu * ones)
        pt.testing.assert_close(actual_2, self.expected_alpha * ones)

    def test_4d(self):
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
        mock = Mock(return_value=pt.ones(1))
        self.custom.activate.forward = mock
        inp = pt.ones(4)
        _ = self.custom(inp)
        actual_mu = mock.call_args_list[0][0][0]
        actual_alpha = mock.call_args_list[1][0][0]
        pt.testing.assert_close(actual_mu, pt.ones(1))
        pt.testing.assert_close(actual_alpha, pt.ones(1))


if __name__ == '__main__':
    unittest.main()
