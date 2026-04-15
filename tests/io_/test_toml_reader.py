import pickle
import unittest
import textwrap
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from tomllib import TOMLDecodeError
from swak.io import TomlReader, Reader, Storage, Mode, NotFound


class TestInstantiation(unittest.TestCase):

    def test_is_reader(self):
        self.assertTrue(issubclass(TomlReader, Reader))

    @patch.object(Reader, '__init__')
    def test_reader_init_called_defaults(self, init):
        _ = TomlReader()
        init.assert_called_once_with(
            '', Storage.FILE, Mode.RB, 32, None, {}, 'raise'
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_custom(self, init):
        _ = TomlReader(
                '/path/to/file.toml',
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                {'toml': 'kwargs'},
                'warn'
        )
        init.assert_called_once_with(
            '/path/to/file.toml',
            Storage.MEMORY,
            Mode.RB,
            16,
            {'storage': 'kws'},
            {'toml': 'kwargs'},
            'warn'
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_enum(self, init):
        _ = TomlReader(
            '/path/to/file.toml',
            Storage.MEMORY,
            16,
            {'storage': 'kws'},
            {'toml': 'kwargs'},
            NotFound.IGNORE,
        )
        init.assert_called_once_with(
            '/path/to/file.toml',
            Storage.MEMORY,
            Mode.RB,
            16,
            {'storage': 'kws'},
            {'toml': 'kwargs'},
            'ignore',
        )


class TestAttributes(unittest.TestCase):

    def test_has_toml_kws(self):
        read = TomlReader()
        self.assertTrue(hasattr(read, 'toml_kws'))

    def test_default_toml_kws(self):
        read = TomlReader()
        self.assertDictEqual({}, read.toml_kws)

    def test_custom_toml_kws(self):
        read = TomlReader(toml_kws={'toml': 'kws'})
        self.assertDictEqual({'toml': 'kws'}, read.toml_kws)

    def test_has_not_found(self):
        read = TomlReader()
        self.assertTrue(hasattr(read, 'not_found'))

    def test_default_not_found(self):
        read = TomlReader()
        self.assertEqual('raise', read.not_found)

    def test_custom_not_found(self):
        read = TomlReader(not_found='warn')
        self.assertEqual('warn', read.not_found)

    def test_wrong_not_found_raises(self):
        with self.assertRaises(ValueError):
            _ = TomlReader(not_found='wrong')


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.storage = Storage.FILE
        self.toml = {
            'foo': 'bar',
            'baz': {'answer': 42},
            'greet': [
                {'name': 'Hello'},
                {'name': 'World'},
            ],
        }
        self.dir = TemporaryDirectory()
        self.file = self.dir.name + '/file.toml'
        self.path = Path(self.file)
        self.content = textwrap.dedent("""
            foo = "bar"

            [baz]
            answer = 42

            [[greet]]
            name = "Hello"

            [[greet]]
            name = "World"
        """).encode()
        with self.path.open('wb') as file:
            file.write(self.content)

    def tearDown(self):
        self.dir.cleanup()

    def test_callable(self):
        read = TomlReader()
        self.assertTrue(callable(read))

    @patch.object(Reader, '_non_root')
    def test_non_root_called_default(self, non_root):
        non_root.return_value = self.file
        read = TomlReader(self.file, self.storage)
        _ = read()
        non_root.assert_called_once_with('')

    @patch.object(Reader, '_non_root')
    def test_non_root_called_custom(self, non_root):
        non_root.return_value = self.file
        read = TomlReader(self.file, self.storage)
        _ = read('/some/other/file.toml')
        non_root.assert_called_once_with('/some/other/file.toml')

    @patch.object(Reader, '_managed')
    def test_managed_called(self, managed):
        read = TomlReader(self.file, self.storage)
        with self.path.open('rb') as file:
            managed.return_value = file
            _ = read()
            managed.assert_called_once_with(self.file)

    @patch.object(Reader, '_managed')
    @patch('swak.io.toml.tomllib.load')
    def test_tomlib_called_defaults(self, load, managed):
        read = TomlReader(self.file, self.storage)
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.toml
            _ = read()
            load.assert_called_once_with(file)

    @patch.object(Reader, '_managed')
    @patch('swak.io.toml.tomllib.load')
    def test_tomlib_called_custom(self, load, managed):
        read = TomlReader(self.file, self.storage, toml_kws={'toml': 'kws'})
        with self.path.open('rt') as file:
            managed.return_value = file
            load.return_value = self.toml
            _ = read()
            load.assert_called_once_with(file, toml='kws')

    def test_raises_on_file_not_found(self):
        read = TomlReader('/some/other/file.toml', self.storage)
        with self.assertRaises(FileNotFoundError):
            _ = read()

    def test_warns_on_file_not_found(self):
        read = TomlReader(
            '/some/other/file.toml',
            self.storage,
            not_found='warn'
        )
        with self.assertWarns(UserWarning):
            _ = read()

    def test_ignores_on_file_not_found(self):
        read = TomlReader(
            '/some/other/file.toml',
            self.storage,
            not_found='ignore'
        )
        actual = read()
        self.assertDictEqual({}, actual)

    def test_invalid_toml_raises(self):
        read = TomlReader(self.file, self.storage)
        invalid = b'invalid = [unclosed'
        with self.path.open('wb') as file:
            file.write(invalid)
        with self.assertRaises(TOMLDecodeError):
            _ = read()

    def test_return_value(self):
        read = TomlReader(self.file, self.storage)
        actual = read()
        self.assertDictEqual(self.toml, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        read = TomlReader()
        expected = ("TomlReader('/', 'file',"
                    " 32.0, {}, {}, 'raise')")
        self.assertEqual(expected, repr(read))

    def test_custom_repr(self):
        read = TomlReader(
                '/path/file.toml',
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                {'toml': 'kwargs'},
                'warn'
        )
        expected = ("TomlReader('/path/file.toml', 'memory', 16.0,"
                    " {'storage': 'kws'}, {'toml': 'kwargs'}, 'warn')")
        self.assertEqual(expected, repr(read))

    def test_pickle_works(self):
        read = TomlReader()
        _ = pickle.loads(pickle.dumps(read))


if __name__ == '__main__':
    unittest.main()
