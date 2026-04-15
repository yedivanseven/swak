import pickle
import unittest
import polars as pl
from unittest.mock import patch, Mock
from tempfile import TemporaryDirectory
from pathlib import Path
import polars.testing
from swak.pl.io import LazyFrame2Parquet, LazyWriter, LazyStorage


class TestInstantiation(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.parquet'

    def test_is_writer(self):
        self.assertTrue(issubclass(LazyFrame2Parquet, LazyWriter))

    @patch.object(LazyWriter, '__init__')
    def test_writer_init_called_defaults(self, init):
        _ = LazyFrame2Parquet(self.path)
        init.assert_called_once_with(
            self.path,
            LazyStorage.FILE,
            None
        )

    @patch.object(LazyWriter, '__init__')
    def test_writer_init_called_custom(self, init):
        _ = LazyFrame2Parquet(
            '/some/other/file.parquet',
            LazyStorage.AZURE,
            {'foo': 'bar'},
            answer=42
        )
        init.assert_called_once_with(
            '/some/other/file.parquet',
            LazyStorage.AZURE,
            {'foo': 'bar'},
            answer=42
        )


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.parquet'

    def test_has_kwargs(self):
        write = LazyFrame2Parquet(self.path)
        self.assertTrue(hasattr(write, 'kwargs'))

    def test_default_parquet_kws(self):
        write = LazyFrame2Parquet(self.path)
        self.assertDictEqual({}, write.kwargs)

    def test_custom_parquet_kws(self):
        write = LazyFrame2Parquet(self.path, answer=42)
        self.assertDictEqual({'answer': 42}, write.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.storage = LazyStorage.FILE
        self.dir = TemporaryDirectory()
        self.file = self.dir.name + '/file.parquet'
        self.path = Path(self.file)
        self.df = Mock()

    def tearDown(self):
        self.dir.cleanup()

    def test_callable(self):
        write = LazyFrame2Parquet(self.file)
        self.assertTrue(callable(write))

    @patch.object(LazyWriter, '_uri_from')
    def test_uri_from_called(self, uri_from):
        uri_from.return_value = self.file
        write = LazyFrame2Parquet(self.file, self.storage)
        _ = write(self.df, 'foo', 42)
        uri_from.assert_called_once_with('foo', 42)

    def test_to_parquet_called_defaults(self):
        write = LazyFrame2Parquet(self.file, self.storage)
        _ = write(self.df)
        self.df.sink_parquet.assert_called_once_with(
            self.file,
            storage_options={}
        )

    def test_to_parquet_called_custom(self):
        write = LazyFrame2Parquet(
            self.file,
            storage=LazyStorage.AZURE,
            storage_kws={'foo': 'bar'},
            answer=42
        )
        _ = write(self.df)
        self.df.sink_parquet.assert_called_once_with(
            'az:/' + self.file,
            storage_options={'foo': 'bar'},
            answer=42
        )

    def test_return_value(self):
        write = LazyFrame2Parquet(self.file, self.storage)
        actual = write(self.df)
        self.assertTupleEqual((), actual)

    def test_actually_saves_polars(self):
        df = pl.DataFrame([{'foo': 42}, {'bar': 43}])
        write = LazyFrame2Parquet(self.file, self.storage)
        _ = write(df.lazy())
        with self.path.open('rb') as file:
            actual = pl.read_parquet(file)
        pl.testing.assert_frame_equal(actual, df)

    def test_subdirectory_created_file(self):
        df = pl.DataFrame([{'foo': 42}, {'bar': 43}])
        path = self.dir.name + '/sub/folder/file.parquet'
        write = LazyFrame2Parquet(path, self.storage, mkdir=True)
        _ = write(df.lazy())
        with Path(path).open('rb') as file:
            actual = pl.read_parquet(file)
        pl.testing.assert_frame_equal(actual, df)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.path = '/path/file.parquet'

    def test_default_repr(self):
        write = LazyFrame2Parquet(self.path)
        expected = "LazyFrame2Parquet('/path/file.parquet', 'file', {})"
        self.assertEqual(expected, repr(write))

    def test_custom_repr(self):
        write = LazyFrame2Parquet(
            self.path,
            'hf',
            storage_kws={'foo': 'bar'},
            answer=42
        )
        expected = ("LazyFrame2Parquet('/path/file.parquet', 'hf', "
                    "{'foo': 'bar'}, answer=42)")
        self.assertEqual(expected, repr(write))

    def test_pickle_works(self):
        write = LazyFrame2Parquet(self.path)
        _ = pickle.loads(pickle.dumps(write))


if __name__ == '__main__':
    unittest.main()
