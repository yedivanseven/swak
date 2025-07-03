import pickle
import unittest
from unittest.mock import patch, mock_open, Mock
from swak.io import DataFrame2Parquet, Writer, Storage, Mode


class TestInstantiation(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.parquet'

    def test_is_writer(self):
        self.assertTrue(issubclass(DataFrame2Parquet, Writer))

    @patch.object(Writer, '__init__')
    def test_writer_init_called_defaults(self, init):
        _ = DataFrame2Parquet(self.path)
        init.assert_called_once_with(
            self.path,
            Storage.FILE,
            False,
            False,
            Mode.WB,
            32,
            None,
            {}
        )

    @patch.object(Writer, '__init__')
    def test_writer_init_called_custom(self, init):
        _ = DataFrame2Parquet(
            '/sone/other/file.csv',
            Storage.MEMORY,
            True,
            True,
            16,
            {'foo': 'bar'},
            {'answer': 42},
        )
        init.assert_called_once_with(
            '/sone/other/file.csv',
            Storage.MEMORY,
            True,
            True,
            Mode.WB,
            16,
            {'foo': 'bar'},
            {'answer': 42}
        )


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.parquet'

    def test_has_parquet_kws(self):
        write = DataFrame2Parquet(self.path)
        self.assertTrue(hasattr(write, 'parquet_kws'))

    def test_default_parquet_kws(self):
        write = DataFrame2Parquet(self.path)
        self.assertDictEqual({}, write.parquet_kws)

    def test_custom_parquet_kws(self):
        write = DataFrame2Parquet(self.path, parquet_kws={'answer': 42})
        self.assertDictEqual({'answer': 42}, write.parquet_kws)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.parquet'
        self.df = Mock()

    def test_callable(self):
        write = DataFrame2Parquet(self.path)
        self.assertTrue(callable(write))

    @patch.object(Writer, '_uri_from')
    def test_uri_from_called(self, uri_from):
        uri_from.return_value = self.path
        write = DataFrame2Parquet(
            self.path,
            storage=Storage.MEMORY,
            overwrite=True
        )
        _ = write(self.df, 'foo', 42)
        uri_from.assert_called_once_with('foo', 42)

    @patch.object(Writer, '_managed')
    @patch.object(Writer, '_uri_from')
    def test_managed_called(self, uri_from, managed):
        uri_from.return_value = 'generated uri'
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        write = DataFrame2Parquet(
            self.path,
            storage=Storage.MEMORY,
            overwrite=True
        )
        value = write(self.df, 'foo', 42)
        managed.assert_called_once_with('generated uri')
        self.assertTupleEqual((), value)

    @patch.object(Writer, '_managed')
    @patch.object(Writer, '_uri_from')
    def test_managed_not_called(self, uri_from, managed):
        uri_from.return_value = ''
        write = DataFrame2Parquet(
            self.path,
            storage=Storage.MEMORY,
            overwrite=True
        )
        value = write(self.df, 'foo', 42)
        managed.assert_not_called()
        self.assertTupleEqual((), value)

    @patch.object(Writer, '_managed')
    def test_to_parquet_called_defaults(self, managed):
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        write = DataFrame2Parquet(
            self.path,
            storage=Storage.MEMORY,
            overwrite=True
        )
        _ = write(self.df)
        self.df.to_parquet.assert_called_once_with(mock_file.return_value)

    @patch.object(Writer, '_managed')
    def test_to_parquet_called_custom(self, managed):
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        write = DataFrame2Parquet(
            self.path,
            storage=Storage.MEMORY,
            overwrite=True,
            parquet_kws={'answer': 42}
        )
        _ = write(self.df)
        self.df.to_parquet.assert_called_once_with(
            mock_file.return_value,
            answer=42
        )

    @patch.object(Writer, '_managed')
    def test_write_parquet_called(self, managed):
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        df = Mock(spec=['write_parquet'])
        write = DataFrame2Parquet(
            self.path,
            storage=Storage.MEMORY,
            overwrite=True
        )
        _ = write(df)
        df.write_parquet.assert_called_once_with(mock_file.return_value)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.path = '/path/file.parquet'

    def test_default_repr(self):
        write = DataFrame2Parquet(self.path)
        expected = ("DataFrame2Parquet('/path/file.parquet', "
                    "'file', False, False, 'wb', 32.0, {}, {})")
        self.assertEqual(expected, repr(write))

    def test_custom_repr(self):
        write = DataFrame2Parquet(self.path, parquet_kws={'answer': 42})
        expected = ("DataFrame2Parquet('/path/file.parquet', "
                    "'file', False, False, 'wb', 32.0, {}, {'answer': 42})")
        self.assertEqual(expected, repr(write))

    def test_pickle_works(self):
        write = DataFrame2Parquet(self.path)
        _ = pickle.loads(pickle.dumps(write))


if __name__ == '__main__':
    unittest.main()
