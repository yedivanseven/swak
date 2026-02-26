import json
import pickle
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
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
            '/some/other/file.json',
            Storage.MEMORY,
            True,
            True,
            16,
            {'foo': 'bar'},
            {'answer': 42},
            False
        )
        init.assert_called_once_with(
            '/some/other/file.json',
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
        self.storage = Storage.FILE
        self.dir = TemporaryDirectory()
        self.file = self.dir.name + '/file.json'
        self.path = Path(self.file)
        self.json = {'hello': 'world', 'foo': {'bar': 'baz'}, 'answer': 42}

    def tearDown(self):
        self.dir.cleanup()

    def test_callable(self):
        write = JsonWriter(self.file)
        self.assertTrue(callable(write))

    @patch.object(Writer, '_uri_from')
    def test_uri_from_called(self, uri_from):
        uri_from.return_value = self.file
        write = JsonWriter(self.file, self.storage)
        _ = write(self.json, 'foo', 42)
        uri_from.assert_called_once_with('foo', 42)

    @patch.object(Writer, '_managed')
    def test_managed_called_default(self, managed):
        with self.path.open('wt') as file:
            managed.return_value = file
            write = JsonWriter(self.file, self.storage, overwrite=True)
            _ = write(self.json, 'foo', 42)
        managed.assert_called_once_with(self.file, None)

    @patch.object(Writer, '_managed')
    def test_managed_called_gzip_suffix(self, managed):
        zipped = self.file + '.gz'
        with Path(zipped).open('wt') as file:
            managed.return_value = file
            write = JsonWriter(zipped, self.storage, overwrite=True)
            _ = write(self.json, 'foo', 42)
        managed.assert_called_once_with(zipped, Compression.GZIP)

    @patch.object(Writer, '_managed')
    def test_managed_called_gzip_false(self, managed):
        zipped = self.file + '.gz'
        with Path(zipped).open('wt') as file:
            managed.return_value = file
            write = JsonWriter(
                zipped,
                storage=self.storage,
                overwrite=True,
                gzip=False
            )
            _ = write(self.json, 'foo', 42)
        managed.assert_called_once_with(zipped, None)

    @patch.object(Writer, '_managed')
    def test_managed_called_gzip_true(self, managed):
        zipped = self.file + '.gz'
        with Path(zipped).open('wt') as file:
            managed.return_value = file
            write = JsonWriter(
                zipped,
                storage=self.storage,
                overwrite=True,
                gzip=True
            )
            _ = write(self.json, 'foo', 42)
        managed.assert_called_once_with(zipped, Compression.GZIP)

    @patch.object(Writer, '_managed')
    @patch.object(Writer, '_uri_from')
    def test_managed_not_called(self, uri_from, managed):
        uri_from.return_value = ''
        write = JsonWriter(self.file, self.storage)
        _ = write(self.json, 'foo', 42)
        managed.assert_not_called()

    @patch('swak.io.json.json.dump')
    @patch.object(Writer, '_managed')
    def test_dump_called_defaults(self, managed, dump):
        with self.path.open('wt') as file:
            managed.return_value = file
            write = JsonWriter(self.file, self.storage, overwrite=True)
            _ = write(self.json)
            dump.assert_called_once_with(self.json, file)

    @patch('swak.io.json.json.dump')
    @patch.object(Writer, '_managed')
    def test_dump_called_custom(self, managed, dump):
        with self.path.open('wt') as file:
            managed.return_value = file
            write = JsonWriter(
                self.file,
                storage=self.storage,
                overwrite=True,
                json_kws={'answer': 42}
            )
            _ = write(self.json)
            dump.assert_called_once_with(
                self.json,
                file,
                answer=42
            )

    def test_return_value(self):
        write = JsonWriter(self.file, self.storage)
        actual = write(self.json)
        self.assertTupleEqual((), actual)

    def test_actually_saves(self):
        write = JsonWriter(self.file, self.storage)
        _ = write(self.json)
        with self.path.open('rt') as file:
            actual = json.load(file)
        self.assertDictEqual(actual, self.json)

    def test_subdirectory_created_file(self):
        path = self.dir.name + '/sub/folder/file.json'
        write = JsonWriter(path, self.storage)
        _ = write(self.json)
        with write.fs.open(path, 'rt') as file:
            actual = json.load(file)
        self.assertDictEqual(actual, self.json)

    def test_subdirectory_created_memory(self):
        path = self.dir.name + '/sub/folder/file.json'
        write = JsonWriter(path, 'memory')
        _ = write(self.json)
        with write.fs.open(path, 'rt') as file:
            actual = json.load(file)
        self.assertDictEqual(actual, self.json)


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
