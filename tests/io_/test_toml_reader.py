import pickle
import unittest
from unittest.mock import patch, mock_open
import textwrap
from tomllib import TOMLDecodeError
from swak.io.types import NotFound
from swak.io import TomlReader, Reader, Storage, Mode


class TestInstantiation(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.toml'

    def test_is_reader(self):
        self.assertTrue(issubclass(TomlReader, Reader))

    @patch.object(Reader, '__init__')
    def test_reader_init_called_defaults(self, init):
        _ = TomlReader(self.path)
        init.assert_called_once_with(
            self.path, Storage.FILE, Mode.RB, 32, None, {}, 'raise'
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_custom(self, init):
        _ = TomlReader(
                self.path,
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                {'toml': 'kwargs'},
                'warn'
        )
        init.assert_called_once_with(
            self.path,
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
            self.path,
            Storage.MEMORY,
            16,
            {'storage': 'kws'},
            {'toml': 'kwargs'},
            NotFound.IGNORE,
        )
        init.assert_called_once_with(
            self.path,
            Storage.MEMORY,
            Mode.RB,
            16,
            {'storage': 'kws'},
            {'toml': 'kwargs'},
            'ignore',
        )


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.toml'

    def test_has_toml_kws(self):
        read = TomlReader(self.path)
        self.assertTrue(hasattr(read, 'toml_kws'))

    def test_default_toml_kws(self):
        read = TomlReader(self.path)
        self.assertDictEqual({}, read.toml_kws)

    def test_custom_toml_kws(self):
        read = TomlReader(self.path, toml_kws={'toml': 'kws'})
        self.assertDictEqual({'toml': 'kws'}, read.toml_kws)

    def test_has_not_found(self):
        read = TomlReader(self.path)
        self.assertTrue(hasattr(read, 'not_found'))

    def test_default_not_found(self):
        read = TomlReader(self.path)
        self.assertEqual('raise', read.not_found)

    def test_custom_not_found(self):
        read = TomlReader(self.path, not_found='warn')
        self.assertEqual('warn', read.not_found)

    def test_wrong_not_found_raises(self):
        with self.assertRaises(ValueError):
            _ = TomlReader(self.path, not_found='wrong')


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.storage = Storage.MEMORY
        self.toml = {
            'foo': 'bar',
            'baz': {'answer': 42},
            'greet': [
                {'name': 'Hello'},
                {'name': 'World'},
            ],
        }
        self.path = '/path/to/file.toml'
        self.content = textwrap.dedent("""
            foo = "bar"

            [baz]
            answer = 42

            [[greet]]
            name = "Hello"

            [[greet]]
            name = "World"
        """).encode()

    def test_callable(self):
        read = TomlReader(self.path)
        self.assertTrue(callable(read))

    @patch.object(Reader, '_non_root')
    def test_non_root_called_default(self, non_root):
        non_root.return_value = self.path
        read = TomlReader('/some/other/file.toml', self.storage)
        with read.fs.open(self.path, 'wb') as file:
            file.write(self.content)
        _ = read()
        non_root.assert_called_once_with('')

    @patch.object(Reader, '_non_root')
    def test_non_root_called_custom(self, non_root):
        non_root.return_value = self.path
        read = TomlReader(self.path, self.storage)
        with read.fs.open(self.path, 'wb') as file:
            file.write(self.content)
        _ = read('/some/other/file.toml')
        non_root.assert_called_once_with('/some/other/file.toml')

    @patch.object(Reader, '_non_root')
    @patch.object(Reader, '_managed')
    def test_managed_called(self, managed, non_root):
        non_root.return_value = self.path
        mock_file = mock_open(read_data=self.content)
        managed.return_value = mock_file.return_value
        read = TomlReader('/some/other/file.toml', self.storage)
        with read.fs.open(self.path, 'wb') as file:
            file.write(self.content)
        _ = read()
        managed.assert_called_once_with(self.path)

    @patch.object(Reader, '_managed')
    @patch('swak.io.toml.tomllib.load')
    def test_tomlib_called_defaults(self, load, managed):
        mock_file = mock_open(read_data=self.content)
        managed.return_value = mock_file.return_value
        read = TomlReader(self.path, self.storage)
        with read.fs.open(self.path, 'wb') as file:
            file.write(self.content)
        _ = read(self.path)
        load.assert_called_once_with(mock_file.return_value)

    @patch.object(Reader, '_managed')
    @patch('swak.io.toml.tomllib.load')
    def test_tomlib_called_custom(self, load, managed):
        mock_file = mock_open(read_data=self.content)
        managed.return_value = mock_file.return_value
        read = TomlReader(self.path, self.storage, toml_kws={'toml': 'kws'})
        with read.fs.open(self.path, 'wb') as file:
            file.write(self.content)
        _ = read(self.path)
        load.assert_called_once_with(mock_file.return_value, toml='kws')

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
        read = TomlReader(self.path, self.storage)
        invalid = b'invalid = [unclosed'
        with read.fs.open(self.path, 'wb') as file:
            file.write(invalid)
        with self.assertRaises(TOMLDecodeError):
            _ = read(self.path)

    def test_return_value(self):
        read = TomlReader(self.path, self.storage)
        with read.fs.open(self.path, 'wb') as file:
            file.write(self.content)
        actual = read(self.path)
        self.assertDictEqual(self.toml, actual)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.path = '/path/file.toml'

    def test_default_repr(self):
        read = TomlReader(self.path)
        expected = ("TomlReader('/path/file.toml', 'file',"
                    " 32.0, {}, {}, 'raise')")
        self.assertEqual(expected, repr(read))

    def test_custom_repr(self):
        read = TomlReader(
                self.path,
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
        read = TomlReader(self.path)
        _ = pickle.loads(pickle.dumps(read))


if __name__ == '__main__':
    unittest.main()
