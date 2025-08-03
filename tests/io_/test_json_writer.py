import pickle
import unittest
from unittest.mock import patch, mock_open
from swak.io import JsonWriter, Writer, Storage, Mode, Compression


class TestInstantiation(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.json'

    def test_is_writer(self):
        self.assertTrue(issubclass(JsonWriter, Writer))

    @patch.object(Writer, '__init__')
    def test_writer_init_called_defaults(self, init):
        _ = JsonWriter(self.path)
        init.assert_called_once_with(
            self.path,
            Storage.FILE,
            False,
            False,
            Mode.WT,
            32,
            None,
            {},
            None
        )

    @patch.object(Writer, '__init__')
    def test_writer_init_called_custom(self, init):
        _ = JsonWriter(
            '/sone/other/file.csv',
            Storage.MEMORY,
            True,
            True,
            16,
            {'foo': 'bar'},
            {'answer': 42},
            False
        )
        init.assert_called_once_with(
            '/sone/other/file.csv',
            Storage.MEMORY,
            True,
            True,
            Mode.WT,
            16,
            {'foo': 'bar'},
            {'answer': 42},
            False
        )


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.json'

    def test_has_gzip(self):
        write = JsonWriter(self.path)
        self.assertTrue(hasattr(write, 'gzip'))

    def test_default_gzip(self):
        write = JsonWriter(self.path)
        self.assertIsNone(write.gzip)

    def test_custom_gzip(self):
        write = JsonWriter(self.path, gzip=True)
        self.assertIsInstance(write.gzip, bool)
        self.assertTrue(write.gzip)

    def test_has_json_kws(self):
        write = JsonWriter(self.path)
        self.assertTrue(hasattr(write, 'json_kws'))

    def test_default_json_kws(self):
        write = JsonWriter(self.path)
        self.assertDictEqual({}, write.json_kws)

    def test_custom_json_kws(self):
        write = JsonWriter(self.path, json_kws={'answer': 42})
        self.assertDictEqual({'answer': 42}, write.json_kws)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.json'
        self.storage = Storage.MEMORY
        self.json = {'hello': 'world', 'foo': {'bar': 'baz'}, 'answer': 42}

    def test_callable(self):
        write = JsonWriter(self.path)
        self.assertTrue(callable(write))

    @patch.object(Writer, '_uri_from')
    def test_uri_from_called(self, uri_from):
        uri_from.return_value = self.path
        write = JsonWriter(
            self.path,
            storage=self.storage,
            overwrite=True
        )
        _ = write(self.json, 'foo', 42)
        uri_from.assert_called_once_with('foo', 42)

    @patch.object(Writer, '_managed')
    @patch.object(Writer, '_uri_from')
    def test_managed_called_default(self, uri_from, managed):
        uri_from.return_value = 'generated uri'
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        write = JsonWriter(
            self.path,
            storage=self.storage,
            overwrite=True
        )
        value = write(self.json, 'foo', 42)
        managed.assert_called_once_with('generated uri', None)
        self.assertTupleEqual((), value)

    @patch.object(Writer, '_managed')
    @patch.object(Writer, '_uri_from')
    def test_managed_called_gzip_suffix(self, uri_from, managed):
        uri_from.return_value = 'generated_uri.gz'
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        write = JsonWriter(
            self.path,
            storage=self.storage,
            overwrite=True
        )
        value = write(self.json, 'foo', 42)
        managed.assert_called_once_with('generated_uri.gz', Compression.GZIP)
        self.assertTupleEqual((), value)

    @patch.object(Writer, '_managed')
    @patch.object(Writer, '_uri_from')
    def test_managed_called_gzip_false(self, uri_from, managed):
        uri_from.return_value = 'generated_uri.gz'
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        write = JsonWriter(
            self.path,
            storage=self.storage,
            overwrite=True,
            gzip=False
        )
        value = write(self.json, 'foo', 42)
        managed.assert_called_once_with('generated_uri.gz', None)
        self.assertTupleEqual((), value)

    @patch.object(Writer, '_managed')
    @patch.object(Writer, '_uri_from')
    def test_managed_called_gzip_true(self, uri_from, managed):
        uri_from.return_value = 'generated uri'
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        write = JsonWriter(
            self.path,
            storage=self.storage,
            overwrite=True,
            gzip=True
        )
        value = write(self.json, 'foo', 42)
        managed.assert_called_once_with('generated uri', Compression.GZIP)
        self.assertTupleEqual((), value)

    @patch.object(Writer, '_managed')
    @patch.object(Writer, '_uri_from')
    def test_managed_not_called(self, uri_from, managed):
        uri_from.return_value = ''
        write = JsonWriter(
            self.path,
            storage=self.storage,
            overwrite=True
        )
        value = write(self.json, 'foo', 42)
        managed.assert_not_called()
        self.assertTupleEqual((), value)

    @patch('swak.io.json.json.dump')
    @patch.object(Writer, '_managed')
    def test_dump_called_defaults(self, managed, dump):
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        write = JsonWriter(
            self.path,
            storage=self.storage,
            overwrite=True
        )
        _ = write(self.json)
        dump.assert_called_once_with(self.json, mock_file.return_value)

    @patch('swak.io.json.json.dump')
    @patch.object(Writer, '_managed')
    def test_dump_called_custom(self, managed, dump):
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        write = JsonWriter(
            self.path,
            storage=self.storage,
            overwrite=True,
            json_kws={'answer': 42}
        )
        _ = write(self.json)
        dump.assert_called_once_with(
            self.json,
            mock_file.return_value,
            answer=42
        )


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.path = '/path/file.json'

    def test_default_repr(self):
        write = JsonWriter(self.path)
        expected = ("JsonWriter('/path/file.json', 'file', "
                    "False, False, 32.0, {}, {}, None)")
        self.assertEqual(expected, repr(write))

    def test_custom_repr(self):
        write = JsonWriter(self.path, gzip=False, json_kws={'answer': 42})
        expected = ("JsonWriter('/path/file.json', 'file', False, "
                    "False, 32.0, {}, {'answer': 42}, False)")
        self.assertEqual(expected, repr(write))

    def test_pickle_works(self):
        write = JsonWriter(self.path)
        _ = pickle.loads(pickle.dumps(write))


if __name__ == '__main__':
    unittest.main()
