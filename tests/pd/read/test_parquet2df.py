import unittest
from unittest.mock import patch
import pandas as pd
from swak.pd.read import LocalParquet2DataFrame, local_parquet_2_dataframe


class TestInstantiation(unittest.TestCase):

    def test_empty(self):
        load = LocalParquet2DataFrame()
        self.assertTrue(hasattr(load, 'kwargs'))
        self.assertDictEqual({}, load.kwargs)

    def test_kwargs(self):
        load = LocalParquet2DataFrame(hello='world', foo='bar')
        self.assertTrue(hasattr(load, 'kwargs'))
        self.assertDictEqual({'hello': 'world', 'foo': 'bar'}, load.kwargs)


class TestCall(unittest.TestCase):

    def test_callable(self):
        load = LocalParquet2DataFrame()
        self.assertTrue(callable(load))

    @patch('pandas.read_parquet')
    def test_function_called(self, method):
        load = LocalParquet2DataFrame()
        _ = load('path')
        method.assert_called_once()
        method.assert_called_once_with('path')

    @patch('pandas.read_parquet')
    def test_function_called_kwargs(self, method):
        load = LocalParquet2DataFrame(foo='bar')
        _ = load('path')
        method.assert_called_once()
        method.assert_called_once_with('path', foo='bar')

    @patch('pandas.read_parquet', return_value=pd.DataFrame([1, 2, 3]))
    def test_function_returns(self, _):
        query = LocalParquet2DataFrame()
        result = query('path')
        pd.testing.assert_frame_equal(pd.DataFrame([1, 2, 3]), result)


class TestFunction(unittest.TestCase):

    @patch('pandas.read_parquet')
    def test_function_called(self, method):
        _ = local_parquet_2_dataframe('path')
        method.assert_called_once()
        method.assert_called_once_with('path')

    @patch('pandas.read_parquet')
    def test_function_called_kwargs(self, method):
        _ = local_parquet_2_dataframe('path', foo='bar')
        method.assert_called_once()
        method.assert_called_once_with('path', foo='bar')

    @patch('pandas.read_parquet', return_value=pd.DataFrame([1, 2, 3]))
    def test_function_returns(self, _):
        result = local_parquet_2_dataframe('path')
        pd.testing.assert_frame_equal(pd.DataFrame([1, 2, 3]), result)


class TestRepr(unittest.TestCase):

    def test_repr(self):
        reader = LocalParquet2DataFrame(a=3.0, b='foo')
        expected = "LocalParquet2DataFrame(a=3.0, b='foo')"
        self.assertEqual(expected, repr(reader))


if __name__ == '__main__':
    unittest.main()
