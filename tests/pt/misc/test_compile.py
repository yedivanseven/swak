import unittest
from unittest.mock import patch, Mock
import torch.nn as ptn
from swak.pt.exceptions import CompileError
from swak.pt.misc import Compile


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


if __name__ == '__main__':
    unittest.main()
