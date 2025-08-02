import pickle
import unittest
from unittest.mock import patch, mock_open
import textwrap
from yaml import Loader, SafeLoader
from yaml.scanner import ScannerError
from swak.io.types import NotFound
from swak.io import YamlReader, Reader, Storage, Mode


class TestInstantiation(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.yml'

    def test_is_reader(self):
        self.assertTrue(issubclass(YamlReader, Reader))

    @patch.object(Reader, '__init__')
    def test_reader_init_called_defaults(self, init):
        _ = YamlReader(self.path)
        init.assert_called_once_with(
            self.path, Storage.FILE, Mode.RB, 32, None, Loader, 'raise'
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_custom(self, init):
        _ = YamlReader(
                self.path,
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                SafeLoader,
                'warn'
        )
        init.assert_called_once_with(
            self.path,
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
            self.path,
            Storage.MEMORY,
            16,
            {'storage': 'kws'},
            SafeLoader,
            NotFound.IGNORE,
        )
        init.assert_called_once_with(
            self.path,
            Storage.MEMORY,
            Mode.RB,
            16,
            {'storage': 'kws'},
            SafeLoader,
            'ignore',
        )


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.yml'

    def test_has_toml_kws(self):
        read = YamlReader(self.path)
        self.assertTrue(hasattr(read, 'loader'))

    def test_default_toml_kws(self):
        read = YamlReader(self.path)
        self.assertIs(Loader, read.loader)

    def test_custom_toml_kws(self):
        read = YamlReader(self.path, loader=SafeLoader)
        self.assertIs(SafeLoader, read.loader)

    def test_has_not_found(self):
        read = YamlReader(self.path)
        self.assertTrue(hasattr(read, 'not_found'))

    def test_default_not_found(self):
        read = YamlReader(self.path)
        self.assertEqual('raise', read.not_found)

    def test_custom_not_found(self):
        read = YamlReader(self.path, not_found='warn')
        self.assertEqual('warn', read.not_found)

    def test_wrong_not_found_raises(self):
        with self.assertRaises(ValueError):
            _ = YamlReader(self.path, not_found='wrong')


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.storage = Storage.MEMORY
        self.yaml = {
            'foo': 'bar',
            'baz': {'answer': 42},
            'greet': [
                {'name': 'Hello'},
                {'name': 'World'},
            ],
        }
        self.path = '/path/to/file.yml'
        self.content = textwrap.dedent("""
            foo: bar
            baz:
              answer: 42
            greet:
              - name: Hello
              - name: World
        """).encode()

    def test_callable(self):
        read = YamlReader(self.path)
        self.assertTrue(callable(read))

    @patch.object(Reader, '_non_root')
    def test_non_root_called_default(self, non_root):
        non_root.return_value = self.path
        read = YamlReader('/some/other/file.yml', self.storage)
        with read.fs.open(self.path, 'wb') as file:
            file.write(self.content)
        _ = read()
        non_root.assert_called_once_with('')

    @patch.object(Reader, '_non_root')
    def test_non_root_called_custom(self, non_root):
        non_root.return_value = self.path
        read = YamlReader(self.path, self.storage)
        with read.fs.open(self.path, 'wb') as file:
            file.write(self.content)
        _ = read('/some/other/file.yml')
        non_root.assert_called_once_with('/some/other/file.yml')

    @patch.object(Reader, '_non_root')
    @patch.object(Reader, '_managed')
    def test_managed_called(self, managed, non_root):
        non_root.return_value = self.path
        mock_file = mock_open(read_data=self.content)
        managed.return_value = mock_file.return_value
        read = YamlReader('/some/other/file.yml', self.storage)
        with read.fs.open(self.path, 'wb') as file:
            file.write(self.content)
        _ = read()
        managed.assert_called_once_with(self.path)

    @patch.object(Reader, '_managed')
    @patch('swak.io.yaml.yaml.load')
    def test_yaml_load_called_defaults(self, load, managed):
        mock_file = mock_open(read_data=self.content)
        managed.return_value = mock_file.return_value
        read = YamlReader(self.path, self.storage)
        with read.fs.open(self.path, 'wb') as file:
            file.write(self.content)
        _ = read(self.path)
        load.assert_called_once_with(mock_file.return_value, Loader)

    @patch.object(Reader, '_managed')
    @patch('swak.io.yaml.yaml.load')
    def test_yaml_load_called_custom(self, load, managed):
        mock_file = mock_open(read_data=self.content)
        managed.return_value = mock_file.return_value
        read = YamlReader(self.path, self.storage, loader=SafeLoader)
        with read.fs.open(self.path, 'wb') as file:
            file.write(self.content)
        _ = read(self.path)
        load.assert_called_once_with(mock_file.return_value, SafeLoader)

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
        read = YamlReader(self.path, self.storage)
        invalid = b'key1: value1\nkey2 value2'
        with read.fs.open(self.path, 'wb') as file:
            file.write(invalid)
        with self.assertRaises(ScannerError):
            _ = read(self.path)

    def test_return_value(self):
        read = YamlReader(self.path, self.storage)
        with read.fs.open(self.path, 'wb') as file:
            file.write(self.content)
        actual = read(self.path)
        self.assertDictEqual(self.yaml, actual)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.path = '/path/file.yml'

    def test_default_repr(self):
        read = YamlReader(self.path)
        expected = ("YamlReader('/path/file.yml', 'file',"
                    " 'rb', 32.0, {}, Loader, 'raise')")
        self.assertEqual(expected, repr(read))

    def test_custom_repr(self):
        read = YamlReader(
                self.path,
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                SafeLoader,
                'warn'
        )
        expected = ("YamlReader('/path/file.yml', 'memory', 'rb', 16.0,"
                    " {'storage': 'kws'}, SafeLoader, 'warn')")
        self.assertEqual(expected, repr(read))

    def test_pickle_works(self):
        read = YamlReader(self.path)
        _ = pickle.loads(pickle.dumps(read))


if __name__ == '__main__':
    unittest.main()
