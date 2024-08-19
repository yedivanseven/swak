import os
import pickle
import unittest
from unittest.mock import patch
import pandas as pd
from swak.pd.read import ParquetReader


class TestAttributes(unittest.TestCase):

    def test_empty(self):
        read = ParquetReader()
        self.assertTrue(hasattr(read, 'base_dir'))
        self.assertEqual(os.getcwd(), read.base_dir)
        self.assertTrue(hasattr(read, 'kwargs'))
        self.assertDictEqual({}, read.kwargs)

    def test_base_dir(self):
        read = ParquetReader('/foo')
        self.assertTrue(hasattr(read, 'base_dir'))
        self.assertEqual('/foo', read.base_dir)

    def test_base_dir_adds_preceding_slash(self):
        read = ParquetReader('foo')
        self.assertTrue(hasattr(read, 'base_dir'))
        self.assertEqual('/foo', read.base_dir)

    def test_base_dir_strips(self):
        read = ParquetReader('/ foo/ ')
        self.assertTrue(hasattr(read, 'base_dir'))
        self.assertEqual('/foo', read.base_dir)

    def test_kwargs_only(self):
        read = ParquetReader(a='bar', b=42)
        self.assertTrue(hasattr(read, 'base_dir'))
        self.assertEqual(os.getcwd(), read.base_dir)
        self.assertTrue(hasattr(read, 'kwargs'))
        self.assertDictEqual({'a': 'bar', 'b': 42}, read.kwargs)

    def test_base_dir_and_kwargs(self):
        read = ParquetReader('foo', a='foo', b=42)
        self.assertTrue(hasattr(read, 'base_dir'))
        self.assertEqual('/foo', read.base_dir)
        self.assertTrue(hasattr(read, 'kwargs'))
        self.assertDictEqual({'a': 'foo', 'b': 42}, read.kwargs)


class TestCall(unittest.TestCase):

    def test_callable(self):
        read = ParquetReader()
        self.assertTrue(callable(read))

    @patch('pandas.read_parquet')
    def test_called_empty(self, mock):
        read = ParquetReader('/foo')
        _ = read()
        mock.assert_called_once()
        mock.assert_called_once_with('/foo/')

    @patch('pandas.read_parquet')
    def test_called_path_file(self, mock):
        read = ParquetReader('/foo')
        _ = read('test.pqt')
        mock.assert_called_once()
        mock.assert_called_once_with('/foo/test.pqt')

    @patch('pandas.read_parquet')
    def test_called_path_dir(self, mock):
        read = ParquetReader('/foo')
        _ = read('bar/')
        mock.assert_called_once()
        mock.assert_called_once_with('/foo/bar/')

    @patch('pandas.read_parquet')
    def test_path_left_stripped(self, mock):
        read = ParquetReader('/foo')
        _ = read(' / bar')
        mock.assert_called_once()
        mock.assert_called_once_with('/foo/bar')

    @patch('pandas.read_parquet')
    def test_path_right_stripped(self, mock):
        read = ParquetReader('/foo')
        _ = read('bar  ')
        mock.assert_called_once()
        mock.assert_called_once_with('/foo/bar')

    @patch('pandas.read_parquet')
    def test_called_kwargs(self, method):
        read = ParquetReader('/foo', a='bar', b=42)
        _ = read('baz')
        method.assert_called_once()
        method.assert_called_once_with('/foo/baz', a='bar', b=42)

    @patch('pandas.read_parquet', return_value=pd.DataFrame([1, 2, 3]))
    def test_returns(self, _):
        query = ParquetReader()
        result = query()
        pd.testing.assert_frame_equal(pd.DataFrame([1, 2, 3]), result)


class TestMisc(unittest.TestCase):

    def test_repr(self):
        reader = ParquetReader('foo', a='bar', b=42)
        expected = "ParquetReader('/foo', a='bar', b=42)"
        self.assertEqual(expected, repr(reader))

    def test_pickle_works(self):
        reader = ParquetReader()
        _ = pickle.dumps(reader)


if __name__ == '__main__':
    unittest.main()
