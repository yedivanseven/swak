import pickle
import unittest
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal as pl_assert_frame_equal
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from swak.io import Csv2DataFrame, Reader, Storage, Mode
from swak.io import Bears


class TestInstantiation(unittest.TestCase):

    def test_is_reader(self):
        self.assertTrue(issubclass(Csv2DataFrame, Reader))

    @patch.object(Reader, '__init__')
    def test_reader_init_called_defaults(self, init):
        _ = Csv2DataFrame()
        init.assert_called_once_with(
            '', Storage.FILE, Mode.RT, 32, None, {}, 'pandas'
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_custom(self, init):
        _ = Csv2DataFrame(
                '/path/to/file.csv',
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                {'csv': 'kws'},
                'polars'
        )
        init.assert_called_once_with(
            '/path/to/file.csv',
            Storage.MEMORY,
            Mode.RT,
            16,
            {'storage': 'kws'},
            {'csv': 'kws'},
            'polars'
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_enum(self, init):
        _ = Csv2DataFrame(
            '/path/to/file.csv',
            Storage.MEMORY,
            16,
            {'storage': 'kws'},
            {'csv': 'kws'},
            Bears.POLARS,
        )
        init.assert_called_once_with(
            '/path/to/file.csv',
            Storage.MEMORY,
            Mode.RT,
            16,
            {'storage': 'kws'},
            {'csv': 'kws'},
            'polars',
        )


class TestAttributes(unittest.TestCase):

    def test_has_csv_kws(self):
        read = Csv2DataFrame()
        self.assertTrue(hasattr(read, 'csv_kws'))

    def test_default_csv_kws(self):
        read = Csv2DataFrame()
        self.assertDictEqual({}, read.csv_kws)

    def test_custom_csv_kws(self):
        read = Csv2DataFrame(csv_kws={'csv': 'kws'})
        self.assertDictEqual({'csv': 'kws'}, read.csv_kws)

    def test_has_bear(self):
        read = Csv2DataFrame()
        self.assertTrue(hasattr(read, 'bear'))

    def test_default_bear(self):
        read = Csv2DataFrame()
        self.assertEqual('pandas', read.bear)

    def test_custom_bear(self):
        read = Csv2DataFrame(bear='polars')
        self.assertEqual('polars', read.bear)

    def test_wrong_bear_raises(self):
        with self.assertRaises(ValueError):
            _ = Csv2DataFrame(bear='grizzly')

    def test_has_read(self):
        read = Csv2DataFrame()
        self.assertTrue(hasattr(read, 'read'))

    def test_default_read(self):
        read = Csv2DataFrame()
        self.assertIs(read.read, pd.read_csv)

    def test_custom_read(self):
        read = Csv2DataFrame(bear='polars')
        self.assertIs(read.read, pl.read_csv)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.storage = Storage.FILE
        self.dir = TemporaryDirectory()
        self.file = self.dir.name + '/file.csv'
        self.path = Path(self.file)
        self.content = [1, 2, 3, 4]
        self.pd_df = pd.DataFrame({'0': self.content})
        self.pl_df = pl.DataFrame({'0': self.content})
        self.pd_df.to_csv(self.file, index=False)

    def tearDown(self):
        self.dir.cleanup()

    def test_callable(self):
        read = Csv2DataFrame()
        self.assertTrue(callable(read))

    @patch.object(Reader, '_non_root')
    def test_non_root_called_default(self, non_root):
        non_root.return_value = self.file
        read = Csv2DataFrame(self.file, self.storage)
        _ = read()
        non_root.assert_called_once_with('')

    @patch.object(Reader, '_non_root')
    def test_non_root_called_custom(self, non_root):
        non_root.return_value = self.file
        read = Csv2DataFrame(self.file, self.storage)
        _ = read('/some/other/file.csv')
        non_root.assert_called_once_with('/some/other/file.csv')

    @patch.object(Reader, '_managed')
    def test_managed_called(self, managed):
        read = Csv2DataFrame(self.file, self.storage)
        with self.path.open('rb') as file:
            managed.return_value = file
            _ = read()
        managed.assert_called_once_with(self.file)

    @patch.object(Reader, '_managed')
    @patch('swak.io.csv.pd.read_csv')
    def test_pandas_read_csv_called_defaults(self, load, managed):
        read = Csv2DataFrame(self.file, self.storage)
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.pd_df
            _ = read()
            load.assert_called_once_with(file)

    @patch.object(Reader, '_managed')
    @patch('swak.io.csv.pd.read_csv')
    def test_pandas_read_csv_called_custom(self, load, managed):
        read = Csv2DataFrame(
            self.file,
            self.storage,
            csv_kws={'csv': 'kws'},
        )
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.pd_df
            _ = read()
            load.assert_called_once_with(file, csv='kws')

    @patch.object(Reader, '_managed')
    @patch('swak.io.csv.pl.read_csv')
    def test_polars_read_csv_called_defaults(self, load, managed):
        read = Csv2DataFrame(self.file, self.storage, bear=Bears.POLARS)
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.pl_df
            _ = read()
            load.assert_called_once_with(file)

    @patch.object(Reader, '_managed')
    @patch('swak.io.csv.pl.read_csv')
    def test_polars_read_csv_called_custom(self, load, managed):
        read = Csv2DataFrame(
            self.file,
            self.storage,
            csv_kws={'csv': 'kws'},
            bear='polars'
        )
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.pl_df
            _ = read()
            load.assert_called_once_with(file, csv='kws')

    def test_raises_on_file_not_found(self):
        read = Csv2DataFrame('/some/other/file.csv', self.storage)
        with self.assertRaises(FileNotFoundError):
            _ = read()

    def test_return_value_pandas(self):
        read = Csv2DataFrame(self.file, self.storage, )
        actual = read()
        pd.testing.assert_frame_equal(actual, self.pd_df)

    def test_return_value_polars(self):
        read = Csv2DataFrame(self.file, self.storage, bear='polars')
        actual = read()
        pl_assert_frame_equal(actual, self.pl_df)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        read = Csv2DataFrame()
        expected = ("Csv2DataFrame('/', 'file',"
                    " 32.0, {}, {}, 'pandas')")
        self.assertEqual(expected, repr(read))

    def test_custom_repr(self):
        read = Csv2DataFrame(
                '/path/file.csv',
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                {'csv': 'kws'},
                'polars'
        )
        expected = ("Csv2DataFrame('/path/file.csv', 'memory', 16.0,"
                    " {'storage': 'kws'}, {'csv': 'kws'}, 'polars')")
        self.assertEqual(expected, repr(read))

    def test_pickle_works(self):
        read = Csv2DataFrame()
        _ = pickle.loads(pickle.dumps(read))


if __name__ == '__main__':
    unittest.main()
