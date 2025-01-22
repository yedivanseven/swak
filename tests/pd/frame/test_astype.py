import pickle
import numpy as np
import pandas as pd
import unittest
from unittest.mock import Mock
from swak.pd import AsType


class TestAttributes(unittest.TestCase):

    def test_arg(self):
        convert = AsType(float)
        self.assertTrue(hasattr(convert, 'types'))
        self.assertIs(float, convert.types)
        self.assertTrue(hasattr(convert, 'kwargs'))
        self.assertDictEqual({}, convert.kwargs)

    def test_arg_kwargs(self):
        convert = AsType(float, copy=True)
        self.assertTrue(hasattr(convert, 'types'))
        self.assertIs(float, convert.types)
        self.assertTrue(hasattr(convert, 'kwargs'))
        self.assertDictEqual({'copy': True}, convert.kwargs)

    def test_dict(self):
        convert = AsType({0: float})
        self.assertTrue(hasattr(convert, 'types'))
        self.assertDictEqual({0: float}, convert.types)
        self.assertTrue(hasattr(convert, 'kwargs'))
        self.assertDictEqual({}, convert.kwargs)

    def test_dict_kwargs(self):
        convert = AsType({0: float}, copy=True)
        self.assertTrue(hasattr(convert, 'types'))
        self.assertDictEqual({0: float}, convert.types)
        self.assertTrue(hasattr(convert, 'kwargs'))
        self.assertDictEqual({'copy': True}, convert.kwargs)


class TestMethodCall(unittest.TestCase):

    def setUp(self) -> None:
        self.mock = Mock()

    def test_callable(self):
        convert = AsType(float)
        self.assertTrue(callable(convert))

    def test_arg(self):
        convert = AsType(float)
        _ = convert(self.mock)
        self.mock.astype.assert_called_once()
        self.mock.astype.assert_called_once_with(float)

    def test_arg_kwargs(self):
        convert = AsType(float, copy=True)
        _ = convert(self.mock)
        self.mock.astype.assert_called_once()
        self.mock.astype.assert_called_once_with(float, copy=True)

    def test_dict(self):
        convert = AsType({0: float})
        _ = convert(self.mock)
        self.mock.astype.assert_called_once()
        self.mock.astype.assert_called_once_with({0: float})

    def test_dict_kwargs(self):
        convert = AsType({0: float}, copy=True)
        _ = convert(self.mock)
        self.mock.astype.assert_called_once()
        self.mock.astype.assert_called_once_with({0: float}, copy=True)


class TestUsage(unittest.TestCase):

    def setUp(self) -> None:
        self.df = pd.DataFrame([[1, 5], [2, 6], [3, 7], [4, 8]])
        self.no_cols = pd.DataFrame()
        self.no_rows = pd.DataFrame(columns=[0, 1])

    def test_entire_frame(self):
        convert = AsType(float)
        converted = convert(self.df)
        first, second = converted.dtypes.to_list()
        self.assertIs(np.dtype('float64'), first)
        self.assertIs(np.dtype('float64'), second)

    def test_one_column(self):
        convert = AsType({0: float})
        converted = convert(self.df)
        first, second = converted.dtypes.to_list()
        self.assertIs(np.dtype('float64'), first)
        self.assertIs(np.dtype('int64'), second)

    def test_all_columns(self):
        convert = AsType({0: str, 1: float})
        converted = convert(self.df)
        first, second = converted.dtypes.to_list()
        self.assertIs(np.dtype('O'), first)
        self.assertIs(np.dtype('float64'), second)

    def test_arg_no_cols(self):
        convert = AsType(float)
        converted = convert(self.no_cols)
        self.assertTrue(converted.empty)

    def test_arg_kwargs_no_cols(self):
        convert = AsType(float, copy=True)
        converted = convert(self.no_cols)
        self.assertTrue(converted.empty)

    def test_arg_no_rows(self):
        convert = AsType(float)
        converted = convert(self.no_rows)
        self.assertTrue(converted.empty)
        self.assertListEqual([0, 1], converted.columns.to_list())
        first, second = converted.dtypes.to_list()
        self.assertIs(np.dtype('float64'), first)
        self.assertIs(np.dtype('float64'), second)

    def test_dict_no_rows_one_column(self):
        convert = AsType({0: float})
        converted = convert(self.no_rows)
        self.assertTrue(converted.empty)
        self.assertListEqual([0, 1], converted.columns.to_list())
        first, second = converted.dtypes.to_list()
        self.assertIs(np.dtype('float64'), first)
        self.assertIs(np.dtype('O'), second)

    def test_dict_no_rows_all_columns(self):
        convert = AsType({0: float, 1: int})
        converted = convert(self.no_rows)
        self.assertTrue(converted.empty)
        self.assertListEqual([0, 1], converted.columns.to_list())
        first, second = converted.dtypes.to_list()
        self.assertIs(np.dtype('float64'), first)
        self.assertIs(np.dtype('int64'), second)


class TestMisc(unittest.TestCase):

    def test_python_type(self):
        convert = AsType(float, copy=False)
        self.assertEqual('AsType(float, copy=False)', repr(convert))

    def test_numpy_type(self):
        convert = AsType(np.float32)
        self.assertEqual('AsType(float32)', repr(convert))

    def test_dict_python_types(self):
        convert = AsType({'a': float, 2: int, 'c': str}, errors='raise')
        expected = "AsType({'a': float, 2: int, 'c': str}, errors='raise')"
        self.assertEqual(expected, repr(convert))

    def test_dict_numpy_types(self):
        convert = AsType({'a': np.float32, 2: np.int8})
        expected = "AsType({'a': float32, 2: int8})"
        self.assertEqual(expected, repr(convert))

    def test_dict_empty(self):
        convert = AsType({}, copy=True)
        expected = "AsType({}, copy=True)"
        self.assertEqual(expected, repr(convert))

    def test_pickle_works(self):
        convert = AsType(float, copy=False)
        _ = pickle.loads(pickle.dumps(convert))


if __name__ == '__main__':
    unittest.main()
