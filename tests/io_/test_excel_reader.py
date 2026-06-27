import pickle
import unittest
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal as pl_assert_frame_equal
from unittest.mock import patch
from pathlib import Path
from python_calamine import CalamineError
from swak.io import Excel2DataFrame, Reader, Storage, Mode
from swak.io import Bears


class TestInstantiation(unittest.TestCase):

    def test_is_reader(self):
        self.assertTrue(issubclass(Excel2DataFrame, Reader))

    @patch.object(Reader, '__init__')
    def test_reader_init_called_defaults(self, init):
        _ = Excel2DataFrame()
        init.assert_called_once_with(
            '',
            Storage.FILE,
            Mode.RB,
            32,
            None,
            {'engine': 'calamine'},
            'pandas'
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_custom(self, init):
        _ = Excel2DataFrame(
                '/path/to/file.excel',
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                {'excel': 'kws'},
                'polars'
        )
        init.assert_called_once_with(
            '/path/to/file.excel',
            Storage.MEMORY,
            Mode.RB,
            16,
            {'storage': 'kws'},
            {'excel': 'kws', 'engine': 'calamine'},
            'polars'
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_enum(self, init):
        _ = Excel2DataFrame(
            '/path/to/file.excel',
            Storage.MEMORY,
            16,
            {'storage': 'kws'},
            {'excel': 'kws'},
            Bears.POLARS,
        )
        init.assert_called_once_with(
            '/path/to/file.excel',
            Storage.MEMORY,
            Mode.RB,
            16,
            {'storage': 'kws'},
            {'excel': 'kws', 'engine': 'calamine'},
            'polars',
        )


class TestAttributes(unittest.TestCase):

    def test_has_excel_kws(self):
        read = Excel2DataFrame()
        self.assertTrue(hasattr(read, 'excel_kws'))

    def test_default_excel_kws(self):
        read = Excel2DataFrame()
        self.assertDictEqual({'engine': 'calamine'}, read.excel_kws)

    def test_custom_excel_kws(self):
        read = Excel2DataFrame(excel_kws={'excel': 'kws'})
        self.assertDictEqual(
            {'excel': 'kws', 'engine': 'calamine'},
            read.excel_kws
        )

    def test_has_bear(self):
        read = Excel2DataFrame()
        self.assertTrue(hasattr(read, 'bear'))

    def test_default_bear(self):
        read = Excel2DataFrame()
        self.assertEqual('pandas', read.bear)

    def test_custom_bear(self):
        read = Excel2DataFrame(bear='polars')
        self.assertEqual('polars', read.bear)

    def test_wrong_bear_raises(self):
        with self.assertRaises(ValueError):
            _ = Excel2DataFrame(bear='grizzly')


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.storage = Storage.FILE
        self.file = str((Path(__file__).parent / 'valid.ods').resolve())
        self.invalid = str((Path(__file__).parent / 'invalid.ods').resolve())
        self.path = Path(self.file)
        self.content = [2, 3, 4]
        self.pd_df = pd.DataFrame({1: self.content})
        self.pl_df = pl.DataFrame({'1': self.content})

    def test_callable(self):
        read = Excel2DataFrame()
        self.assertTrue(callable(read))

    @patch.object(Reader, '_non_root')
    def test_non_root_called_default(self, non_root):
        non_root.return_value = self.file
        read = Excel2DataFrame(self.file, self.storage)
        _ = read()
        non_root.assert_called_once_with('')

    @patch.object(Reader, '_non_root')
    def test_non_root_called_custom(self, non_root):
        non_root.return_value = self.file
        read = Excel2DataFrame(self.file, self.storage)
        _ = read('/some/other/file.xlsx')
        non_root.assert_called_once_with('/some/other/file.xlsx')

    @patch.object(Reader, '_managed')
    def test_managed_called(self, managed):
        read = Excel2DataFrame(self.file, self.storage)
        with self.path.open('rb') as file:
            managed.return_value = file
            _ = read()
        managed.assert_called_once_with(self.file)

    @patch.object(Reader, '_managed')
    @patch('swak.io.excel.pd.read_excel')
    def test_pandas_read_excel_called_defaults(self, load, managed):
        read = Excel2DataFrame(self.file, self.storage)
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.pd_df
            _ = read()
            load.assert_called_once_with(file, engine='calamine')

    @patch.object(Reader, '_managed')
    @patch('swak.io.excel.pd.read_excel')
    def test_pandas_read_excel_called_custom(self, load, managed):
        read = Excel2DataFrame(
            self.file,
            self.storage,
            excel_kws={'excel': 'kws'},
        )
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.pd_df
            _ = read()
            load.assert_called_once_with(file, engine='calamine', excel='kws')



    @patch('swak.io.excel.BytesIO')
    @patch.object(Reader, '_managed')
    @patch('swak.io.excel.pl.read_excel')
    def test_polars_read_excel_called_defaults(self, load, managed, bytes_io):
        read = Excel2DataFrame(self.file, self.storage, bear=Bears.POLARS)
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.pl_df
            bytes_io.return_value = 42
            _ = read()
            load.assert_called_once_with(42, engine='calamine')

    @patch('swak.io.excel.BytesIO')
    @patch.object(Reader, '_managed')
    @patch('swak.io.excel.pl.read_excel')
    def test_polars_read_excel_called_custom(self, load, managed, bytes_io):
        read = Excel2DataFrame(
            self.file,
            self.storage,
            excel_kws={'excel': 'kws'},
            bear='polars'
        )
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.pl_df
            bytes_io.return_value = 42
            _ = read()
            load.assert_called_once_with(42, engine='calamine', excel='kws')

    def test_raises_on_file_not_found(self):
        read = Excel2DataFrame('/some/other/file.excel', self.storage)
        with self.assertRaises(FileNotFoundError):
            _ = read()

    def test_invalid_excel_raises(self):
        read = Excel2DataFrame(self.invalid, self.storage)
        with self.assertRaises(CalamineError):
            _ = read()

    def test_return_value_pandas(self):
        read = Excel2DataFrame(self.file, self.storage)
        actual = read()
        pd.testing.assert_frame_equal(actual, self.pd_df)

    def test_return_value_polars(self):
        read = Excel2DataFrame(self.file, self.storage, bear='polars')
        actual = read()
        pl_assert_frame_equal(actual, self.pl_df)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        read = Excel2DataFrame()
        expected = ("Excel2DataFrame('/', 'file',"
                    " 32.0, {}, {'engine': 'calamine'}, 'pandas')")
        self.assertEqual(expected, repr(read))

    def test_custom_repr(self):
        read = Excel2DataFrame(
                '/path/file.excel',
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                {'excel': 'kws'},
                'polars'
        )
        expected = ("Excel2DataFrame('/path/file.excel', 'memory', 16.0,"
                    " {'storage': 'kws'}, {'engine': 'calamine', "
                    "'excel': 'kws'}, 'polars')")
        self.assertEqual(expected, repr(read))

    def test_pickle_works(self):
        read = Excel2DataFrame()
        _ = pickle.loads(pickle.dumps(read))


if __name__ == '__main__':
    unittest.main()
