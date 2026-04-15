import pickle
import unittest
import textwrap
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from yaml import Loader, SafeLoader
from yaml.scanner import ScannerError
from swak.io import YamlReader, Reader, Storage, Mode, NotFound


class TestInstantiation(unittest.TestCase):

    def test_is_reader(self):
        self.assertTrue(issubclass(YamlReader, Reader))

    @patch.object(Reader, '__init__')
    def test_reader_init_called_defaults(self, init):
        _ = YamlReader()
        init.assert_called_once_with(
            '', Storage.FILE, Mode.RB, 32, None, Loader, 'raise'
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_custom(self, init):
        _ = YamlReader(
            '/path/to/file.yml',
            Storage.MEMORY,
            16,
            {'storage': 'kws'},
            SafeLoader,
            'warn'
        )
        init.assert_called_once_with(
            '/path/to/file.yml',
            Storage.MEMORY,
            Mode.RB,
            16,
            {'storage': 'kws'},
            SafeLoader,
            'warn'
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_enum(self, init):
        _ = YamlReader(
            '/path/to/file.yml',
            Storage.MEMORY,
            16,
            {'storage': 'kws'},
            SafeLoader,
            NotFound.IGNORE,
        )
        init.assert_called_once_with(
            '/path/to/file.yml',
            Storage.MEMORY,
            Mode.RB,
            16,
            {'storage': 'kws'},
            SafeLoader,
            'ignore',
        )


class TestAttributes(unittest.TestCase):

    def test_has_loader(self):
        read = YamlReader()
        self.assertTrue(hasattr(read, 'loader'))

    def test_default_loader(self):
        read = YamlReader()
        self.assertIs(Loader, read.loader)

    def test_custom_loader(self):
        read = YamlReader(loader=SafeLoader)
        self.assertIs(SafeLoader, read.loader)

    def test_has_not_found(self):
        read = YamlReader()
        self.assertTrue(hasattr(read, 'not_found'))

    def test_default_not_found(self):
        read = YamlReader()
        self.assertEqual('raise', read.not_found)

    def test_custom_not_found(self):
        read = YamlReader(not_found='warn')
        self.assertEqual('warn', read.not_found)

    def test_wrong_not_found_raises(self):
        with self.assertRaises(ValueError):
            _ = YamlReader(not_found='wrong')


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.storage = Storage.FILE
        self.yaml = {
            'foo': 'bar',
            'baz': {'answer': 42},
            'greet': [
                {'name': 'Hello'},
                {'name': 'World'},
            ],
        }
        self.dir = TemporaryDirectory()
        self.file = self.dir.name + '/file.yml'
        self.path = Path(self.file)
        self.content = textwrap.dedent("""
            foo: bar
            baz:
              answer: 42
            greet:
              - name: Hello
              - name: World
        """).encode()
        with self.path.open('wb') as file:
            file.write(self.content)

    def tearDown(self):
        self.dir.cleanup()

    def test_callable(self):
        read = YamlReader()
        self.assertTrue(callable(read))

    @patch.object(Reader, '_non_root')
    def test_non_root_called_default(self, non_root):
        non_root.return_value = self.file
        read = YamlReader(self.file, self.storage)
        _ = read()
        non_root.assert_called_once_with('')

    @patch.object(Reader, '_non_root')
    def test_non_root_called_custom(self, non_root):
        non_root.return_value = self.file
        read = YamlReader(self.file, self.storage)
        _ = read('/some/other/file.yml')
        non_root.assert_called_once_with('/some/other/file.yml')

    @patch.object(Reader, '_managed')
    def test_managed_called(self, managed):
        read = YamlReader(self.file, self.storage)
        with self.path.open('rb') as file:
            managed.return_value = file
            _ = read()
            managed.assert_called_once_with(self.file)

    @patch.object(Reader, '_managed')
    @patch('swak.io.yaml.yaml.load')
    def test_yaml_load_called_defaults(self, load, managed):
        read = YamlReader(self.file, self.storage)
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.yaml
            _ = read()
            load.assert_called_once_with(file, Loader)

    @patch.object(Reader, '_managed')
    @patch('swak.io.yaml.yaml.load')
    def test_yaml_load_called_custom(self, load, managed):
        read = YamlReader(self.file, self.storage, loader=SafeLoader)
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.yaml
            _ = read()
            load.assert_called_once_with(file, SafeLoader)

    def test_raises_on_file_not_found(self):
        read = YamlReader('/some/other/file.yml', self.storage)
        with self.assertRaises(FileNotFoundError):
            _ = read()

    def test_warns_on_file_not_found(self):
        read = YamlReader(
            '/some/other/file.yml',
            self.storage,
            not_found='warn'
        )
        with self.assertWarns(UserWarning):
            _ = read()

    def test_ignores_on_file_not_found(self):
        read = YamlReader(
            '/some/other/file.yml',
            self.storage,
            not_found='ignore'
        )
        actual = read()
        self.assertDictEqual({}, actual)

    def test_invalid_yaml_raises(self):
        read = YamlReader(self.file, self.storage)
        invalid = b'key1: value1\nkey2 value2'
        with self.path.open('wb') as file:
            file.write(invalid)
        with self.assertRaises(ScannerError):
            _ = read()

    def test_return_value(self):
        read = YamlReader(self.file, self.storage)
        actual = read(self.file)
        self.assertDictEqual(self.yaml, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        read = YamlReader()
        expected = ("YamlReader('/', 'file',"
                    " 32.0, {}, Loader, 'raise')")
        self.assertEqual(expected, repr(read))

    def test_custom_repr(self):
        read = YamlReader(
                '/path/file.yml',
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                SafeLoader,
                'warn'
        )
        expected = ("YamlReader('/path/file.yml', 'memory', 16.0,"
                    " {'storage': 'kws'}, SafeLoader, 'warn')")
        self.assertEqual(expected, repr(read))

    def test_pickle_works(self):
        read = YamlReader()
        _ = pickle.loads(pickle.dumps(read))


if __name__ == '__main__':
    unittest.main()
