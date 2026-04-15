import pickle
import unittest
import textwrap
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from json import JSONDecodeError
from swak.io import JsonReader, Reader, Storage, Mode, Compression, NotFound


class TestInstantiation(unittest.TestCase):

    def test_is_reader(self):
        self.assertTrue(issubclass(JsonReader, Reader))

    @patch.object(Reader, '__init__')
    def test_reader_init_called_defaults(self, init):
        _ = JsonReader()
        init.assert_called_once_with(
            '', Storage.FILE, Mode.RT, 32, None, {}, 'raise', None
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_custom(self, init):
        _ = JsonReader(
                '/root',
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                {'json': 'kws'},
                'warn',
                True
        )
        init.assert_called_once_with(
            '/root',
            Storage.MEMORY,
            Mode.RT,
            16,
            {'storage': 'kws'},
            {'json': 'kws'},
            'warn',
            True
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_enum(self, init):
        _ = JsonReader(
            '/path/to/file.json',
            Storage.MEMORY,
            16,
            {'storage': 'kws'},
            {'json': 'kws'},
            NotFound.IGNORE,
            False
        )
        init.assert_called_once_with(
            '/path/to/file.json',
            Storage.MEMORY,
            Mode.RT,
            16,
            {'storage': 'kws'},
            {'json': 'kws'},
            'ignore',
            False
        )


class TestAttributes(unittest.TestCase):

    def test_has_json_kws(self):
        read = JsonReader()
        self.assertTrue(hasattr(read, 'json_kws'))

    def test_default_json_kws(self):
        read = JsonReader()
        self.assertDictEqual({}, read.json_kws)

    def test_custom_json_kws(self):
        read = JsonReader(json_kws={'json': 'kws'})
        self.assertDictEqual({'json': 'kws'}, read.json_kws)

    def test_has_not_found(self):
        read = JsonReader()
        self.assertTrue(hasattr(read, 'not_found'))

    def test_default_not_found(self):
        read = JsonReader()
        self.assertEqual('raise', read.not_found)

    def test_custom_not_found(self):
        read = JsonReader(not_found='warn')
        self.assertEqual('warn', read.not_found)

    def test_wrong_not_found_raises(self):
        with self.assertRaises(ValueError):
            _ = JsonReader(not_found='wrong')

    def test_has_gzip(self):
        read = JsonReader()
        self.assertTrue(hasattr(read, 'gzip'))

    def test_default_gzip(self):
        read = JsonReader()
        self.assertIsNone(read.gzip)

    def test_custom_gzip(self):
        read = JsonReader(gzip=True)
        self.assertIsInstance(read.gzip, bool)
        self.assertTrue(read.gzip)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.storage = Storage.FILE
        self.json = {
            'foo': 'bar',
            'baz': {'answer': 42},
            'greet': [
                {'name': 'Hello'},
                {'name': 'World'},
            ],
        }
        self.dir = TemporaryDirectory()
        self.file = self.dir.name + '/file.json'
        self.path = Path(self.file)
        self.content = textwrap.dedent("""
            {
                "foo": "bar",
                "baz": {"answer": 42},
                "greet": [
                    {"name": "Hello"},
                    {"name": "World"}
                ]
            }
        """)
        with self.path.open('wt') as file:
            file.write(self.content)

    def tearDown(self):
        self.dir.cleanup()

    def test_callable(self):
        read = JsonReader()
        self.assertTrue(callable(read))

    @patch.object(Reader, '_non_root')
    def test_non_root_called_default(self, non_root):
        non_root.return_value = self.file
        read = JsonReader(self.file, self.storage)
        _ = read()
        non_root.assert_called_once_with('')

    @patch.object(Reader, '_non_root')
    def test_non_root_called_custom(self, non_root):
        non_root.return_value = self.file
        read = JsonReader(self.file, self.storage)
        _ = read('/some/other/file.json')
        non_root.assert_called_once_with('/some/other/file.json')

    @patch.object(Reader, '_managed')
    def test_managed_called_default(self, managed):
        read = JsonReader(self.file, self.storage)
        with self.path.open('rt') as file:
            managed.return_value = file
            _ = read()
            managed.assert_called_once_with(self.file, None)

    @patch.object(Reader, '_managed')
    def test_managed_called_custom(self, managed):
        read = JsonReader(self.file, self.storage, gzip=True)
        with self.path.open('rt') as file:
            managed.return_value = file
            _ = read()
            managed.assert_called_once_with(self.file, Compression.GZIP)

    @patch.object(Reader, '_managed')
    @patch('swak.io.json.json.load')
    def test_json_load_called_defaults(self, load, managed):
        read = JsonReader(self.file, self.storage)
        with self.path.open('rt') as file:
            managed.return_value = file
            load.return_value = self.json
            _ = read()
            load.assert_called_once_with(file)

    @patch.object(Reader, '_managed')
    @patch('swak.io.json.json.load')
    def test_json_load_called_custom(self, load, managed):
        read = JsonReader(self.file, self.storage, json_kws={'json': 'kws'})
        with self.path.open('rt') as file:
            managed.return_value = file
            load.return_value = self.json
            _ = read()
            load.assert_called_once_with(file, json='kws')

    def test_raises_on_file_not_found(self):
        read = JsonReader('/some/other/file.json', self.storage)
        with self.assertRaises(FileNotFoundError):
            _ = read()

    def test_warns_on_file_not_found(self):
        read = JsonReader(
            '/some/other/file.json',
            self.storage,
            not_found='warn'
        )
        with self.assertWarns(UserWarning):
            _ = read()

    def test_ignores_on_file_not_found(self):
        read = JsonReader(
            '/some/other/file.json',
            self.storage,
            not_found='ignore'
        )
        actual = read()
        self.assertDictEqual({}, actual)

    def test_invalid_json_raises(self):
        read = JsonReader(self.file, self.storage)
        invalid = "{'key': 'value'}"
        with self.path.open('wt') as file:
            file.write(invalid)
        with self.assertRaises(JSONDecodeError):
            _ = read()

    def test_return_value(self):
        read = JsonReader(self.file, self.storage)
        actual = read()
        self.assertDictEqual(self.json, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        read = JsonReader()
        expected = ("JsonReader('/', 'file',"
                    " 32.0, {}, {}, 'raise', None)")
        self.assertEqual(expected, repr(read))

    def test_custom_repr(self):
        read = JsonReader(
                '/path/file.json',
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                {'json': 'kws'},
                'warn',
                False
        )
        expected = ("JsonReader('/path/file.json', 'memory', 16.0,"
                    " {'storage': 'kws'}, {'json': 'kws'}, 'warn', False)")
        self.assertEqual(expected, repr(read))

    def test_pickle_works(self):
        read = JsonReader()
        _ = pickle.loads(pickle.dumps(read))


if __name__ == '__main__':
    unittest.main()
