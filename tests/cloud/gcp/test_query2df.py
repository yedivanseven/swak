import unittest
from unittest.mock import patch
import pickle
import pandas as pd
from swak.cloud.gcp import GbqQuery2DataFrame


class TestAttributes(unittest.TestCase):

    def test_empty(self):
        query = GbqQuery2DataFrame()
        self.assertTrue(hasattr(query, 'kwargs'))
        self.assertDictEqual({}, query.kwargs)

    def test_kwargs(self):
        query = GbqQuery2DataFrame(hello='world', foo='bar')
        self.assertTrue(hasattr(query, 'kwargs'))
        self.assertDictEqual({'hello': 'world', 'foo': 'bar'}, query.kwargs)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        query = GbqQuery2DataFrame()
        self.assertTrue(callable(query))

    @patch('pandas_gbq.read_gbq')
    def test_function_called(self, mock):
        query = GbqQuery2DataFrame()
        _ = query('hello world')
        mock.assert_called_once()
        mock.assert_called_once_with('hello world')

    @patch('pandas_gbq.read_gbq')
    def test_function_called_with_query(self, mock):
        query = GbqQuery2DataFrame()
        _ = query('hello world')
        mock.assert_called_once_with('hello world')

    @patch('pandas_gbq.read_gbq')
    def test_function_called_with_interpolated_query(self, mock):
        query = GbqQuery2DataFrame()
        _ = query('hello {}', 'world')
        mock.assert_called_once_with('hello world')

    @patch('pandas_gbq.read_gbq')
    def test_function_called_kwargs(self, method):
        query = GbqQuery2DataFrame(foo='bar')
        _ = query('hello world')
        method.assert_called_once_with('hello world', foo='bar')

    @patch('pandas_gbq.read_gbq', return_value=pd.DataFrame([1, 2, 3]))
    def test_function_returns(self, _):
        query = GbqQuery2DataFrame()
        result = query('hello world')
        pd.testing.assert_frame_equal(pd.DataFrame([1, 2, 3]), result)


class TestMisc(unittest.TestCase):

    def test_repr_empty(self):
        reader = GbqQuery2DataFrame()
        expected = "GbqQuery2DataFrame()"
        self.assertEqual(expected, repr(reader))

    def test_repr_kwargs(self):
        reader = GbqQuery2DataFrame(a=3.0, b='foo')
        expected = "GbqQuery2DataFrame(a=3.0, b='foo')"
        self.assertEqual(expected, repr(reader))

    def test_pickle_works(self):
        reader = GbqQuery2DataFrame(a=3.0, b='foo')
        _ = pickle.dumps(reader)


if __name__ == '__main__':
    unittest.main()
