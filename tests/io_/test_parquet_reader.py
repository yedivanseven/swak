import pickle
import unittest
import pandas as pd
import polars as pl
from pyarrow import ArrowInvalid
from io import BytesIO
from unittest.mock import patch, Mock
from swak.misc import Bears
from swak.io import Parquet2DataFrame, Reader, Storage, Mode


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
        self.storage = Storage.MEMORY
        self.path = '/path/to/file.parquet'
        self.pd_df = pd.DataFrame([1, 2, 3, 4])
        self.pl_df = pl.DataFrame([1, 2, 3, 4])

    def test_callable(self):
        read = Parquet2DataFrame()
        self.assertTrue(callable(read))

    @patch.object(Reader, '_non_root')
    def test_non_root_called_default(self, non_root):
        non_root.return_value = self.path
        read = Parquet2DataFrame('/some/other/file.parquet', self.storage)
        with read.fs.open(self.path, 'wb') as file:
            self.pd_df.to_parquet(file)
        _ = read()
        non_root.assert_called_once_with('')

    @patch.object(Reader, '_non_root')
    def test_non_root_called_custom(self, non_root):
        non_root.return_value = self.path
        read = Parquet2DataFrame(self.path, self.storage)
        with read.fs.open(self.path, 'wb') as file:
            self.pd_df.to_parquet(file)
        _ = read('/some/other/file.parquet')
        non_root.assert_called_once_with('/some/other/file.parquet')

    @patch.object(Reader, '_non_root')
    @patch.object(Reader, '_managed')
    def test_managed_called(self, managed, non_root):
        non_root.return_value = self.path

        parquet = BytesIO()
        self.pd_df.to_parquet(parquet)
        parquet.seek(0)

        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=parquet)
        mock_context.__exit__ = Mock(return_value=None)
        managed.return_value = mock_context

        read = Parquet2DataFrame('/some/other/file.parquet', self.storage)
        _ = read()
        managed.assert_called_once_with(self.path)

    @patch.object(Reader, '_managed')
    @patch('swak.io.parquet.pd.read_parquet')
    def test_pandas_read_parquet_called_defaults(self, load, managed):
        parquet = BytesIO()
        self.pd_df.to_parquet(parquet)
        parquet.seek(0)

        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=parquet)
        mock_context.__exit__ = Mock(return_value=None)
        managed.return_value = mock_context

        read = Parquet2DataFrame(self.path, self.storage)

        _ = read(self.path)
        load.assert_called_once_with(parquet)

    @patch.object(Reader, '_managed')
    @patch('swak.io.parquet.pd.read_parquet')
    def test_pandas_read_parquet_called_custom(self, load, managed):
        parquet = BytesIO()
        self.pd_df.to_parquet(parquet)
        parquet.seek(0)

        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=parquet)
        mock_context.__exit__ = Mock(return_value=None)
        managed.return_value = mock_context

        read = Parquet2DataFrame(
            self.path,
            self.storage,
            parquet_kws={'parquet': 'kws'},
        )

        _ = read(self.path)
        load.assert_called_once_with(parquet, parquet='kws')

    @patch.object(Reader, '_managed')
    @patch('swak.io.parquet.pl.read_parquet')
    def test_polars_read_parquet_called_defaults(self, load, managed):
        parquet = BytesIO()
        self.pd_df.to_parquet(parquet)
        parquet.seek(0)

        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=parquet)
        mock_context.__exit__ = Mock(return_value=None)
        managed.return_value = mock_context

        read = Parquet2DataFrame(self.path, self.storage, bear='polars')

        _ = read(self.path)
        load.assert_called_once_with(parquet)

    @patch.object(Reader, '_managed')
    @patch('swak.io.parquet.pl.read_parquet')
    def test_polars_read_parquet_called_custom(self, load, managed):
        parquet = BytesIO()
        self.pd_df.to_parquet(parquet)
        parquet.seek(0)

        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=parquet)
        mock_context.__exit__ = Mock(return_value=None)
        managed.return_value = mock_context

        read = Parquet2DataFrame(
            self.path,
            self.storage,
            parquet_kws={'parquet': 'kws'},
            bear='polars'
        )

        _ = read(self.path)
        load.assert_called_once_with(parquet, parquet='kws')

    def test_raises_on_file_not_found(self):
        read = Parquet2DataFrame('/some/other/file.parquet', self.storage)
        with self.assertRaises(FileNotFoundError):
            _ = read()

    def test_invalid_yaml_raises(self):
        read = Parquet2DataFrame(self.path, self.storage)
        invalid = b'not a parquet'
        with read.fs.open(self.path, 'wb') as file:
            file.write(invalid)
        with self.assertRaises(ArrowInvalid):
            _ = read(self.path)

    def test_return_value(self):
        read = Parquet2DataFrame(self.path, self.storage)
        with read.fs.open(self.path, 'wb') as file:
            self.pd_df.to_parquet(file)
        actual = read(self.path)
        pd.testing.assert_frame_equal(actual, self.pd_df)


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
