import pickle
import unittest
import polars as pl
import polars.testing
from polars.exceptions import ComputeError
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from swak.pl.io import LazyReader, Parquet2LazyFrame, LazyStorage


class TestInstantiation(unittest.TestCase):

    def test_is_reader(self):
        self.assertTrue(issubclass(Parquet2LazyFrame, LazyReader))

    @patch.object(LazyReader, '__init__')
    def test_reader_init_called_defaults(self, init):
        _ = Parquet2LazyFrame()
        init.assert_called_once_with('', LazyStorage.FILE, None)

    @patch.object(LazyReader, '__init__')
    def test_reader_init_called_custom(self, init):
        _ = Parquet2LazyFrame(
                '/path/to/file.parquet',
                'memory',
                {'storage': 'kws'},
                answer=42
        )
        init.assert_called_once_with(
            '/path/to/file.parquet',
            'memory',
            {'storage': 'kws'},
            answer=42
        )

    @patch.object(LazyReader, '__init__')
    def test_reader_init_called_enum(self, init):
        _ = Parquet2LazyFrame(
            '/path/to/file.parquet',
            LazyStorage.HF,
            {'storage': 'kws'},
            answer=42
        )
        init.assert_called_once_with(
            '/path/to/file.parquet',
            LazyStorage.HF,
            {'storage': 'kws'},
            answer=42
        )


class TestAttributes(unittest.TestCase):

    def test_has_kargs(self):
        read = Parquet2LazyFrame()
        self.assertTrue(hasattr(read, 'kwargs'))

    def test_default_parquet_kws(self):
        read = Parquet2LazyFrame()
        self.assertDictEqual({}, read.kwargs)

    def test_custom_parquet_kws(self):
        read = Parquet2LazyFrame(answer=42)
        self.assertDictEqual({'answer': 42}, read.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.storage = LazyStorage.FILE
        self.dir = TemporaryDirectory()
        self.file = self.dir.name + '/file.parquet'
        self.path = Path(self.file)
        self.content = [1, 2, 3, 4]
        self.pl_df = pl.DataFrame({'0': self.content})
        self.pl_df.write_parquet(self.file)

    def tearDown(self):
        self.dir.cleanup()

    def test_callable(self):
        read = Parquet2LazyFrame()
        self.assertTrue(callable(read))

    @patch.object(LazyReader, '_non_root')
    def test_non_root_called_default(self, non_root):
        non_root.return_value = self.file
        read = Parquet2LazyFrame(self.file, self.storage)
        _ = read()
        non_root.assert_called_once_with('')

    @patch.object(LazyReader, '_non_root')
    def test_non_root_called_custom(self, non_root):
        non_root.return_value = self.file
        read = Parquet2LazyFrame(self.file, self.storage)
        _ = read('/some/other/file.parquet')
        non_root.assert_called_once_with('/some/other/file.parquet')

    @patch('swak.pl.io.parquet.pl.scan_parquet')
    def test_scan_parquet_called_defaults(self, scan):
        read = Parquet2LazyFrame(self.file, self.storage)
        _ = read()
        scan.assert_called_once_with(self.file, storage_options={})

    @patch('swak.pl.io.parquet.pl.scan_parquet')
    def test_scan_parquet_called_custom(self, scan):
        read = Parquet2LazyFrame(
            self.file,
            LazyStorage.HF,
            storage_kws={'storage': 'kws'},
            answer=42,
        )
        _ = read()
        scan.assert_called_once_with(
            'hf:/' + self.file,
            storage_options={'storage': 'kws'},
            answer=42
        )

    def test_raises_on_file_not_found(self):
        read = Parquet2LazyFrame('/some/other/file.parquet', self.storage)
        lf = read()
        with self.assertRaises(OSError):
            lf.collect()

    def test_invalid_parquet_raises(self):
        read = Parquet2LazyFrame(self.file, self.storage)
        invalid = b'not a parquet'
        with self.path.open('wb') as file:
            file.write(invalid)
        lf = read()
        with self.assertRaises(ComputeError):
            lf.collect()

    def test_return_value_polars(self):
        read = Parquet2LazyFrame(self.file, self.storage)
        actual = read().collect()
        pl.testing.assert_frame_equal(actual, self.pl_df)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        read = Parquet2LazyFrame()
        expected = "Parquet2LazyFrame('/', 'file', {})"
        self.assertEqual(expected, repr(read))

    def test_custom_repr(self):
        read = Parquet2LazyFrame(
                '/path/file.parquet',
                LazyStorage.AZURE,
                {'storage': 'kws'},
                answer=42
        )
        expected = ("Parquet2LazyFrame('/path/file.parquet', 'az',"
                    " {'storage': 'kws'}, answer=42)")
        self.assertEqual(expected, repr(read))

    def test_pickle_works(self):
        read = Parquet2LazyFrame()
        _ = pickle.loads(pickle.dumps(read))


if __name__ == '__main__':
    unittest.main()
