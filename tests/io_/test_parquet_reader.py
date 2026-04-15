import pickle
import unittest
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal as pl_assert_frame_equal
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from pyarrow import ArrowInvalid
from swak.io import Parquet2DataFrame, Reader, Storage, Mode
from swak.io import Bears


class TestInstantiation(unittest.TestCase):

    def test_is_reader(self):
        self.assertTrue(issubclass(Parquet2DataFrame, Reader))

    @patch.object(Reader, '__init__')
    def test_reader_init_called_defaults(self, init):
        _ = Parquet2DataFrame()
        init.assert_called_once_with(
            '', Storage.FILE, Mode.RB, 32, None, {}, 'pandas'
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_custom(self, init):
        _ = Parquet2DataFrame(
                '/path/to/file.parquet',
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                {'parquet': 'kws'},
                'polars'
        )
        init.assert_called_once_with(
            '/path/to/file.parquet',
            Storage.MEMORY,
            Mode.RB,
            16,
            {'storage': 'kws'},
            {'parquet': 'kws'},
            'polars'
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_enum(self, init):
        _ = Parquet2DataFrame(
            '/path/to/file.parquet',
            Storage.MEMORY,
            16,
            {'storage': 'kws'},
            {'parquet': 'kws'},
            Bears.POLARS,
        )
        init.assert_called_once_with(
            '/path/to/file.parquet',
            Storage.MEMORY,
            Mode.RB,
            16,
            {'storage': 'kws'},
            {'parquet': 'kws'},
            'polars',
        )


class TestAttributes(unittest.TestCase):

    def test_has_parquet_kws(self):
        read = Parquet2DataFrame()
        self.assertTrue(hasattr(read, 'parquet_kws'))

    def test_default_parquet_kws(self):
        read = Parquet2DataFrame()
        self.assertDictEqual({}, read.parquet_kws)

    def test_custom_parquet_kws(self):
        read = Parquet2DataFrame(parquet_kws={'parquet': 'kws'})
        self.assertDictEqual({'parquet': 'kws'}, read.parquet_kws)

    def test_has_bear(self):
        read = Parquet2DataFrame()
        self.assertTrue(hasattr(read, 'bear'))

    def test_default_bear(self):
        read = Parquet2DataFrame()
        self.assertEqual('pandas', read.bear)

    def test_custom_bear(self):
        read = Parquet2DataFrame(bear='polars')
        self.assertEqual('polars', read.bear)

    def test_wrong_bear_raises(self):
        with self.assertRaises(ValueError):
            _ = Parquet2DataFrame(bear='grizzly')

    def test_has_read(self):
        read = Parquet2DataFrame()
        self.assertTrue(hasattr(read, 'read'))

    def test_default_read(self):
        read = Parquet2DataFrame()
        self.assertIs(read.read, pd.read_parquet)

    def test_custom_read(self):
        read = Parquet2DataFrame(bear='polars')
        self.assertIs(read.read, pl.read_parquet)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.storage = Storage.FILE
        self.dir = TemporaryDirectory()
        self.file = self.dir.name + '/file.parquet'
        self.path = Path(self.file)
        self.content = [1, 2, 3, 4]
        self.pd_df = pd.DataFrame(self.content)
        self.pl_df = pl.DataFrame({'0': self.content})
        self.pd_df.to_parquet(self.file)

    def tearDown(self):
        self.dir.cleanup()

    def test_callable(self):
        read = Parquet2DataFrame()
        self.assertTrue(callable(read))

    @patch.object(Reader, '_non_root')
    def test_non_root_called_default(self, non_root):
        non_root.return_value = self.file
        read = Parquet2DataFrame(self.file, self.storage)
        _ = read()
        non_root.assert_called_once_with('')

    @patch.object(Reader, '_non_root')
    def test_non_root_called_custom(self, non_root):
        non_root.return_value = self.file
        read = Parquet2DataFrame(self.file, self.storage)
        _ = read('/some/other/file.parquet')
        non_root.assert_called_once_with('/some/other/file.parquet')

    @patch.object(Reader, '_managed')
    def test_managed_called(self, managed):
        read = Parquet2DataFrame(self.file, self.storage)
        with self.path.open('rb') as file:
            managed.return_value = file
            _ = read()
        managed.assert_called_once_with(self.file)

    @patch.object(Reader, '_managed')
    @patch('swak.io.parquet.pd.read_parquet')
    def test_pandas_read_parquet_called_defaults(self, load, managed):
        read = Parquet2DataFrame(self.file, self.storage)
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.pd_df
            _ = read()
            load.assert_called_once_with(file)

    @patch.object(Reader, '_managed')
    @patch('swak.io.parquet.pd.read_parquet')
    def test_pandas_read_parquet_called_custom(self, load, managed):
        read = Parquet2DataFrame(
            self.file,
            self.storage,
            parquet_kws={'parquet': 'kws'},
        )
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.pd_df
            _ = read()
            load.assert_called_once_with(file, parquet='kws')

    @patch.object(Reader, '_managed')
    @patch('swak.io.parquet.pl.read_parquet')
    def test_polars_read_parquet_called_defaults(self, load, managed):
        read = Parquet2DataFrame(self.file, self.storage, bear=Bears.POLARS)
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.pl_df
            _ = read()
            load.assert_called_once_with(file)

    @patch.object(Reader, '_managed')
    @patch('swak.io.parquet.pl.read_parquet')
    def test_polars_read_parquet_called_custom(self, load, managed):
        read = Parquet2DataFrame(
            self.file,
            self.storage,
            parquet_kws={'parquet': 'kws'},
            bear='polars'
        )
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.pl_df
            _ = read()
            load.assert_called_once_with(file, parquet='kws')

    def test_raises_on_file_not_found(self):
        read = Parquet2DataFrame('/some/other/file.parquet', self.storage)
        with self.assertRaises(FileNotFoundError):
            _ = read()

    def test_invalid_parquet_raises(self):
        read = Parquet2DataFrame(self.file, self.storage)
        invalid = b'not a parquet'
        with self.path.open('wb') as file:
            file.write(invalid)
        with self.assertRaises(ArrowInvalid):
            _ = read()

    def test_return_value_pandas(self):
        read = Parquet2DataFrame(self.file, self.storage)
        actual = read()
        pd.testing.assert_frame_equal(actual, self.pd_df)

    def test_return_value_polars(self):
        read = Parquet2DataFrame(self.file, self.storage, bear='polars')
        actual = read()
        pl_assert_frame_equal(actual, self.pl_df)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        read = Parquet2DataFrame()
        expected = ("Parquet2DataFrame('/', 'file',"
                    " 32.0, {}, {}, 'pandas')")
        self.assertEqual(expected, repr(read))

    def test_custom_repr(self):
        read = Parquet2DataFrame(
                '/path/file.parquet',
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                {'parquet': 'kws'},
                'polars'
        )
        expected = ("Parquet2DataFrame('/path/file.parquet', 'memory', 16.0,"
                    " {'storage': 'kws'}, {'parquet': 'kws'}, 'polars')")
        self.assertEqual(expected, repr(read))

    def test_pickle_works(self):
        read = Parquet2DataFrame()
        _ = pickle.loads(pickle.dumps(read))


if __name__ == '__main__':
    unittest.main()
