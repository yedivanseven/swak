import pickle
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import pandas as pd
from swak.pd.write import ParquetWriter


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path'
        self.write = ParquetWriter(self.path)

    def test_has_path(self):
        self.assertTrue(hasattr(self.write, 'path'))

    def test_path(self):
        self.assertEqual(self.path, self.write.path)

    def test_path_like(self):
        write = ParquetWriter(Path(self.path))
        self.assertEqual(self.path, write.path)

    def test_path_stripped(self):
        write = ParquetWriter(f'  {self.path} ')
        self.assertEqual(self.path, write.path)

    def test_has_create(self):
        self.assertTrue(hasattr(self.write, 'create'))

    def test_create(self):
        self.assertIsInstance(self.write.create, bool)
        self.assertFalse(self.write.create)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.write, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.write.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.write = ParquetWriter('/path',True, answer=42)

    def test_create(self):
        self.assertIsInstance(self.write.create, bool)
        self.assertTrue(self.write.create)

    def test_kwargs(self):
        self.assertDictEqual({'answer': 42}, self.write.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        with NamedTemporaryFile() as file:
            self.path = file.name
        self.kwargs = {'answer': 42}
        self.write = ParquetWriter(self.path, **self.kwargs)
        self.df = pd.DataFrame([1, 2, 3])

    def test_callable(self):
        self.assertTrue(callable(self.write))

    @patch('pathlib.Path.mkdir')
    @patch('pandas.DataFrame.to_parquet')
    def test_mkdir_not_called(self, _, mkdir):
        _ = self.write(self.df)
        mkdir.assert_not_called()

    @patch('pathlib.Path.mkdir')
    @patch('pandas.DataFrame.to_parquet')
    def test_mkdir_called(self, _, mkdir):
        write = ParquetWriter(self.path, create=True)
        _ = write(self.df)
        mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('pathlib.Path.mkdir')
    @patch('pandas.DataFrame.to_parquet')
    @patch('swak.pd.write.Path')
    def test_path_interpolated(self, path, *_):
        write = ParquetWriter(' Hello {}, the answer is {}!  ')
        _ = write(self.df, 'there', '42')
        expected = 'Hello there, the answer is 42!'
        path.assert_called_once_with(expected)

    @patch('pathlib.Path.mkdir')
    @patch('pandas.DataFrame.to_parquet')
    def test_to_parquet_called(self, to_parquet, _):
        _ = self.write(self.df)
        to_parquet.assert_called_once_with(self.path, **self.kwargs)

    @patch('pathlib.Path.mkdir')
    @patch('pandas.DataFrame.to_parquet')
    def test_return_value(self, _, __):
        actual = self.write(self.df)
        self.assertTupleEqual((), actual)

    def test_write_raises_without_create(self):
        with TemporaryDirectory() as folder:
            path = folder + '/path/does/not/exist/test.yaml'
            write = ParquetWriter(path)
            with self.assertRaises(OSError):
                _ = write(self.df)

    def test_write_works_with_create(self):
        with TemporaryDirectory() as folder:
            path = folder + '/path/does/not/exist/test.yaml'
            write = ParquetWriter(path, create=True)
            _ = write(self.df)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        reader = ParquetWriter('foo')
        expected = "ParquetWriter('foo', False)"
        self.assertEqual(expected, repr(reader))

    def test_custom_repr(self):
        reader = ParquetWriter('foo', True, a='bar', b=42)
        expected = "ParquetWriter('foo', True, a='bar', b=42)"
        self.assertEqual(expected, repr(reader))

    def test_pickle_works(self):
        reader = ParquetWriter()
        _ = pickle.loads(pickle.dumps(reader))


if __name__ == '__main__':
    unittest.main()
