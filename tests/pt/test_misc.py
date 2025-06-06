import unittest
from unittest.mock import patch, Mock
import torch as pt
import torch.nn as ptn
from swak.pt.exceptions import CompileError, ShapeError, ValidationErrors
from swak.pt.misc import (
    identity,
    Identity,
    Finalizer,
    NegativeBinomialFinalizer,
    Compile,
    Cat,
    LazyCatDim0
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


class TestCompile(unittest.TestCase):

    def setUp(self):
        self.default = Compile()
        self.model = ptn.Linear(1, 1)
        self.inplace = False
        self.disable = True
        self.custom = Compile(self.inplace, self.model, disable=self.disable)
        self.kwargs = {'disable': self.disable}

    def test_default_has_inplace(self):
        self.assertTrue(hasattr(self.default, 'inplace'))

    def test_default_inplace(self):
        self.assertIsInstance(self.default.inplace, bool)
        self.assertTrue(self.default.inplace)

    def test_default_has_model(self):
        self.assertTrue(hasattr(self.default, 'model'))

    def test_default_model(self):
        self.assertIsNone(self.default.model)

    def test_default_has_kwargs(self):
        self.assertTrue(hasattr(self.default, 'kwargs'))

    def test_default_kwargs(self):
        self.assertDictEqual({}, self.default.kwargs)

    def test_custom_inplace(self):
        self.assertIsInstance(self.custom.inplace, bool)
        self.assertIs(self.custom.inplace, self.inplace)

    def test_custom_model(self):
        self.assertIs(self.custom.model, self.model)

    def test_custom_kwargs(self):
        self.assertDictEqual(self.kwargs, self.custom.kwargs)

    def test_callable(self):
        self.assertTrue(callable(self.default))

    def test_raised_on_no_model(self):
        with self.assertRaises(CompileError):
            _ = self.default()

    def test_module_compile_called(self):
        model = Mock()
        actual = self.default(model)
        model.compile.assert_called_once_with()
        self.assertIs(model, actual)

    def test_module_compile_called_kwargs(self):
        model = Mock()
        actual = self.default(model, **self.kwargs)
        model.compile.assert_called_once_with(**self.kwargs)
        self.assertIs(model, actual)

    @patch('torch.compile', return_value=42)
    def test_function_compile_called(self, func):
        actual = self.custom()
        func.assert_called_once_with(self.model, **self.kwargs)
        self.assertEqual(42, actual)

    @patch('torch.compile', return_value=42)
    def test_function_compile_called_kwargs(self, func):
        actual = self.custom(disable=False, hello='world')
        func.assert_called_once_with(self.model, disable=False, hello='world')
        self.assertEqual(42, actual)

    @patch('torch.compile', return_value=42)
    def test_call_model_trumps_instantiation_model(self, func):
        actual = self.custom('Hello world!')
        func.assert_called_once_with('Hello world!', **self.kwargs)
        self.assertEqual(42, actual)

    def test_default_repr(self):
        expected = 'Compile(True, None)'
        self.assertEqual(expected, repr(self.default))

    def test_custom_repr(self):
        expected = 'Compile(False, model, disable=True)'
        self.assertEqual(expected, repr(self.custom))



class TestCat(unittest.TestCase):

    def test_has_dim(self):
        cat = Cat()
        self.assertTrue(hasattr(cat, 'dim'))

    def test_default_dim(self):
        cat = Cat()
        self.assertIsInstance(cat.dim, int)
        self.assertEqual(0, cat.dim)

    def test_custom_dim(self):
        cat = Cat(2)
        self.assertEqual(2, cat.dim)

    def test_callable(self):
        self.assertTrue(callable(Cat()))

    @patch('torch.cat')
    def test_cat_called_with_default_dim(self,  mock):
        cat = Cat()
        _ = cat([1, 2, 3])
        mock.assert_called_once_with([1, 2, 3], dim=0)

    @patch('torch.cat')
    def test_cat_called_with_custom_dim(self,  mock):
        cat = Cat(2)
        _ = cat([1, 2, 3])
        mock.assert_called_once_with([1, 2, 3], dim=2)

    def test_default_repr(self):
        expected = 'Cat(0)'
        self.assertEqual(expected, repr(Cat()))

    def test_custom_repr(self):
        expected = 'Cat(2)'
        self.assertEqual(expected, repr(Cat(2)))


class TestLazyCatDim0(unittest.TestCase):

    def setUp(self):
        two = pt.rand(2, 6)
        three = pt.rand(3, 6)
        self.device = two.device
        self.expected = pt.cat([two, three], dim=0)
        self.cat = LazyCatDim0([two, three])
        two_1d = pt.rand(2)
        three_1d = pt.rand(3)
        self.expected_1d = pt.cat([two_1d, three_1d], dim=0)
        self.cat_1d = LazyCatDim0([two_1d, three_1d])
        two_3d = pt.rand(2, 6, 9)
        three_3d = pt.rand(3, 6, 9)
        self.expected_3d = pt.cat([two_3d, three_3d], dim=0)
        self.cat_3d = LazyCatDim0([two_3d, three_3d])

    def test_empty_iterable_raises(self):
        with self.assertRaises(ShapeError):
            _ = LazyCatDim0([])

    def test_scalar_tensors_raise(self):
        tensors = pt.ones(10), pt.tensor(2.0)
        with self.assertRaises(ValidationErrors):
            _ = LazyCatDim0(tensors)

    def test_different_shapes_raise(self):
        tensors = pt.ones(2, 5), pt.ones(10)
        with self.assertRaises(ValidationErrors):
            _ = LazyCatDim0(tensors)

    @unittest.skipUnless(pt.cuda.is_available(), 'CUDA is not available.')
    def test_different_devices_raise(self):
        tensors = pt.ones(2, 5, device='cpu'), pt.ones(3, 5, device='cuda')
        with self.assertRaises(ValidationErrors):
            _ = LazyCatDim0(tensors)

    def test_different_types_raise(self):
        tensors = pt.ones(2, 5), pt.ones(3, 5, dtype=pt.long)
        with self.assertRaises(ValidationErrors):
            _ = LazyCatDim0(tensors)

    def test_has_lookup(self):
        self.assertTrue(hasattr(self.cat, 'lookup'))

    def test_lookup(self):
        self.assertIsInstance(self.cat.lookup, tuple)
        expected = (0, 0), (0, 1), (1, 0), (1, 1), (1, 2)
        self.assertTupleEqual(expected, self.cat.lookup)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.cat, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.cat.dtype, pt.float)

    def test_has_device(self):
        self.assertTrue(hasattr(self.cat, 'device'))

    def test_device(self):
        self.assertEqual(self.device, self.cat.device)

    def test_has_shape(self):
        self.assertTrue(hasattr(self.cat, 'shape'))

    def test_shape(self):
        expected = pt.Size((5, 6))
        self.assertEqual(expected, self.cat.shape)

    def test_has_size(self):
        self.assertTrue(hasattr(self.cat, 'size'))

    def test_size_callable(self):
        self.assertTrue(callable(self.cat.size))

    def test_call_size(self):
        expected = pt.Size((5, 6))
        self.assertEqual(expected, self.cat.size())

    def test_call_size_arg(self):
        self.assertEqual(5, self.cat.size(0))
        self.assertEqual(6, self.cat.size(1))

    def test_has_to(self):
        self.assertTrue(hasattr(self.cat, 'to'))

    def test_to_callable(self):
        self.assertTrue(callable(self.cat.to))

    def test_call_to(self):
        cast = self.cat.to(pt.long)
        self.assertIs(cast.dtype, pt.long)

    def test_repr(self):
        expected = 'LazyCatDim0(n=5)'
        self.assertEqual(expected, repr(self.cat))

    def test_len(self):
        self.assertEqual(5, len(self.cat))

    def test_iter(self):
        counter = 0
        for tensor in self.cat:
            self.assertEqual(pt.Size([6]), tensor.shape)
            counter += 1
        self.assertEqual(5, counter)

    def test_contains_true(self):
        cat = LazyCatDim0([pt.tensor([1, 2, 3])])
        self.assertTrue(2 in cat)

    def test_contains_false(self):
        cat = LazyCatDim0([pt.tensor([1, 2, 3])])
        self.assertFalse(4 in cat)

    def test_getitem_int(self):
        pt.testing.assert_close(self.cat[0], self.expected[0])
        pt.testing.assert_close(self.cat[1], self.expected[1])
        pt.testing.assert_close(self.cat[2], self.expected[2])
        pt.testing.assert_close(self.cat[3], self.expected[3])
        pt.testing.assert_close(self.cat[4], self.expected[4])

    def test_getitem_slice_step_1(self):
        pt.testing.assert_close(self.cat[1:-1], self.expected[1:-1])

    def test_getitem_slice_step_2(self):
        pt.testing.assert_close(self.cat[1:-1:2], self.expected[1:-1:2])

    def test_getitem_list(self):
        pt.testing.assert_close(self.cat[[1, 2, 4]], self.expected[[1, 2, 4]])

    def test_getitem_empty_tuple(self):
        pt.testing.assert_close(self.cat[()], self.expected[()])

    def test_getitem_one_tuple_int(self):
        pt.testing.assert_close(self.cat[2,], self.expected[2,])

    def test_getitem_one_tuple_slice(self):
        pt.testing.assert_close(self.cat[2:4,], self.expected[2:4,])

    def test_getitem_tuple_int(self):
        pt.testing.assert_close(self.cat[2, 4], self.expected[2, 4])

    def test_getitem_tuple_slice_int(self):
        pt.testing.assert_close(self.cat[1:4, 3], self.expected[1:4, 3])

    def test_getitem_tuple_int_slice(self):
        pt.testing.assert_close(self.cat[3, 1:4], self.expected[3, 1:4])

    def test_getitem_tuple_slice_slice(self):
        pt.testing.assert_close(self.cat[1:3, 2:4], self.expected[1:3, 2:4])

    def test_getitem_int_1d(self):
        pt.testing.assert_close(self.cat_1d[2], self.expected_1d[2])

    def test_getitem_slice_step_1_1d(self):
        pt.testing.assert_close(self.cat_1d[1:-1], self.expected_1d[1:-1])

    def test_getitem_slice_step_2_1d(self):
        pt.testing.assert_close(self.cat_1d[1:-1:2], self.expected_1d[1:-1:2])

    def test_getitem_list_1d(self):
        pt.testing.assert_close(
            self.cat_1d[[1, 2, 4]],
            self.expected_1d[[1, 2, 4]]
        )

    def test_getitem_empty_tuple_1d(self):
        pt.testing.assert_close(self.cat_1d[()], self.expected_1d[()])

    def test_getitem_one_tuple_int_1d(self):
        pt.testing.assert_close(self.cat_1d[2,], self.expected_1d[2,])

    def test_getitem_one_tuple_slice_1d(self):
        pt.testing.assert_close(self.cat_1d[2:4, ], self.expected_1d[2:4, ])

    def test_getitem_int_3d(self):
        pt.testing.assert_close(self.cat_3d[2], self.expected_3d[2])

    def test_getitem_slice_step_1_3d(self):
        pt.testing.assert_close(self.cat_3d[1:-1], self.expected_3d[1:-1])

    def test_getitem_slice_step_2_3d(self):
        pt.testing.assert_close(self.cat_3d[1:-1:2], self.expected_3d[1:-1:2])

    def test_getitem_list_3d(self):
        pt.testing.assert_close(
            self.cat_3d[[1, 2, 4]],
            self.expected_3d[[1, 2, 4]]
        )

    def test_getitem_empty_tuple_3d(self):
        pt.testing.assert_close(self.cat_3d[()], self.expected_3d[()])

    def test_getitem_one_tuple_int_3d(self):
        pt.testing.assert_close(self.cat_3d[2,], self.expected_3d[2,])

    def test_getitem_one_tuple_slice_3d(self):
        pt.testing.assert_close(self.cat_3d[2:4, ], self.expected_3d[2:4, ])

    def test_getitem_tuple_int_3d(self):
        pt.testing.assert_close(self.cat_3d[2, 4], self.expected_3d[2, 4])
        pt.testing.assert_close(
            self.cat_3d[2, 4, 6],
            self.expected_3d[2, 4, 6]
        )

    def test_getitem_tuple_slice_int_3d(self):
        pt.testing.assert_close(self.cat_3d[1:4, 3], self.expected_3d[1:4, 3])
        pt.testing.assert_close(
            self.cat_3d[1:4, 3, 6],
            self.expected_3d[1:4, 3, 6]
        )
        pt.testing.assert_close(
            self.cat_3d[1:4, 3, 2:6],
            self.expected_3d[1:4, 3, 2:6]
        )

    def test_getitem_tuple_int_slice_3d(self):
        pt.testing.assert_close(self.cat_3d[3, 1:4], self.expected_3d[3, 1:4])
        pt.testing.assert_close(
            self.cat_3d[3, 1:4, 6],
            self.expected_3d[3, 1:4, 6]
        )
        pt.testing.assert_close(
            self.cat_3d[3, 1:4, 2:6],
            self.expected_3d[3, 1:4, 2:6]
        )

    def test_getitem_tuple_slice_slice_3d(self):
        pt.testing.assert_close(
            self.cat_3d[1:3, 2:4],
            self.expected_3d[1:3, 2:4]
        )
        pt.testing.assert_close(
            self.cat_3d[1:3, 2:4, 6],
            self.expected_3d[1:3, 2:4, 6]
        )
        pt.testing.assert_close(
            self.cat_3d[1:3, 2:4, 2:6],
            self.expected_3d[1:3, 2:4, 2:6]
        )

if __name__ == '__main__':
    unittest.main()
