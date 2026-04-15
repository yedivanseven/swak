import pickle
import numpy as np
import pandas as pd
import unittest
from unittest.mock import Mock
from swak.pd import AsType


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.convert = AsType(float)

    def test_hast_types(self):
        self.assertTrue(hasattr(self.convert, 'types'))

    def test_types(self):
        self.assertIs(self.convert.types, float)

    def test_has_errors(self):
        self.assertTrue(hasattr(self.convert, 'errors'))

    def test_errors(self):
        self.assertEqual('raise', self.convert.errors)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.types = {'foo': float, 'bar': int}
        self.errors = 'ignore'
        self.convert = AsType(self.types, self.errors)

    def test_types(self):
        self.assertDictEqual(self.types, self.convert.types)

    def test_errors(self):
        self.assertEqual(self.errors, self.convert.errors)


class TestMethodCall(unittest.TestCase):

    def setUp(self):
        self.types = {'foo': float, 'bar': int}
        self.errors = 'ignore'
        self.convert = AsType(self.types, self.errors)

    def test_callable(self):
        convert = AsType(float)
        self.assertTrue(callable(convert))

    def test_dataframe(self):
        df = pd.DataFrame(range(10))
        df.astype = Mock(return_value='answer')
        actual = self.convert(df)
        df.astype.assert_called_once_with(
            self.convert.types,
            errors=self.errors
        )
        self.assertEqual('answer', actual)

    def test_series(self):
        df = pd.Series(range(10))
        df.astype = Mock(return_value='answer')
        actual = self.convert(df)
        df.astype.assert_called_once_with(
            self.convert.types,
            errors=self.errors
        )
        self.assertEqual('answer', actual)


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

    @unittest.skipIf(pd.__version__.startswith('2'), 'Needs pandas >= 3')
    def test_all_columns_pandas_3(self):
        convert = AsType({0: str, 1: float})
        converted = convert(self.df)
        first, second = converted.dtypes.to_list()
        self.assertIsInstance(first, pd.StringDtype)
        self.assertIs(np.dtype('float64'), second)

    @unittest.skipUnless(pd.__version__.startswith('2'), 'Needs pandas < 3')
    def test_all_columns_pandas_2(self):
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
        convert = AsType(float, errors='raise')
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
        convert = AsType(float)
        self.assertEqual("AsType(float, errors='raise')", repr(convert))

    def test_numpy_type(self):
        convert = AsType(np.float32)
        self.assertEqual("AsType(float32, errors='raise')", repr(convert))

    def test_dict_python_types(self):
        convert = AsType({'a': float, 2: int, 'c': str}, errors='ignore')
        expected = "AsType({'a': float, 2: int, 'c': str}, errors='ignore')"
        self.assertEqual(expected, repr(convert))

    def test_dict_numpy_types(self):
        convert = AsType({'a': np.float32, 2: np.int8})
        expected = "AsType({'a': float32, 2: int8}, errors='raise')"
        self.assertEqual(expected, repr(convert))

    def test_dict_empty(self):
        convert = AsType({})
        expected = "AsType({}, errors='raise')"
        self.assertEqual(expected, repr(convert))

    def test_pickle_works(self):
        convert = AsType(float)
        _ = pickle.loads(pickle.dumps(convert))


if __name__ == '__main__':
    unittest.main()
