import pickle
import unittest
from unittest.mock import patch, Mock, mock_open
from pathlib import Path
from swak.text import TomlReader, NotFound


def f(x: str) -> float:
    return float(x)


class TestAttributes(unittest.TestCase):

    def test_empty(self):
        t = TomlReader()
        self.assertTrue(hasattr(t, 'path'))
        self.assertEqual('', t.path)

    def test_path(self):
        t = TomlReader('/hello')
        self.assertTrue(hasattr(t, 'path'))
        self.assertEqual('/hello', t.path)

    def test_path_cast(self):
        t = TomlReader(123)
        self.assertEqual('123', t.path)

    def test_path_like(self):
        t = TomlReader(Path('123'))
        self.assertEqual('123', t.path)

    def test_path_stripped(self):
        t = TomlReader(' hello  ')
        self.assertEqual('hello', t.path)

    def test_default_not_found(self):
        t = TomlReader('hello')
        self.assertTrue(hasattr(t, 'not_found'))
        self.assertEqual(t.not_found, NotFound.RAISE)

    def test_custom_not_found(self):
        t = TomlReader('hello', NotFound.WARN)
        self.assertTrue(hasattr(t, 'not_found'))
        self.assertEqual(t.not_found, NotFound.WARN)

    def test_default_parse_float(self):
        t = TomlReader('hello')
        self.assertTrue(hasattr(t, 'parse_float'))
        self.assertIs(t.parse_float, float)

    def test_custom_parse_float(self):
        t = TomlReader('hello', parse_float=f)
        self.assertTrue(hasattr(t, 'parse_float'))
        self.assertIs(t.parse_float, f)

    def test_default_kwargs(self):
        t = TomlReader('hello')
        self.assertTrue(hasattr(t, 'kwargs'))
        self.assertDictEqual({}, t.kwargs)

    def test_custom_kwargs(self):
        t = TomlReader('hello', world=42)
        self.assertTrue(hasattr(t, 'kwargs'))
        self.assertDictEqual({'world': 42}, t.kwargs)

    def test_mode_kwarg_purged(self):
        t = TomlReader('hello', world=42, mode='w+')
        self.assertTrue(hasattr(t, 'kwargs'))
        self.assertDictEqual({'world': 42}, t.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.path = Path(__file__).parent
        self.dir = str(self.path)

    def test_callable(self):
        t = TomlReader('')
        self.assertTrue(callable(t))

    def test_open_called(self):
        mock = mock_open(read_data=b'')
        with patch('swak.text.read.Path.open', mock):
            t = TomlReader(self.dir)
            _ = t('foo/bar.toml')
        mock.assert_called_once()

    def test_open_called_with_defaults(self):
        mock = mock_open(read_data=b'')
        with patch('swak.text.read.Path.open', mock):
            t = TomlReader(self.dir)
            _ = t('foo/bar.toml')
        mock.assert_called_once_with('rb')

    def test_open_called_with_kwargs(self):
        mock = mock_open(read_data=b'')
        with patch('swak.text.read.Path.open', mock):
            t = TomlReader(self.dir, encoding='utf-8')
            _ = t('foo/bar.toml')
        mock.assert_called_once_with('rb', encoding='utf-8')

    def test_open_called_with_mode_purged(self):
        mock = mock_open(read_data=b'')
        with patch('swak.text.read.Path.open', mock):
            t = TomlReader(self.dir, encoding='utf-8', mode='w+')
            _ = t('foo/bar.toml')
        mock.assert_called_once_with('rb', encoding='utf-8')

    @patch('swak.text.read.tomllib.load')
    def test_load_called(self, mock):
        t = TomlReader(self.dir)
        _ = t('foo/bar.toml')
        mock.assert_called_once()

    @patch('swak.text.read.Path.open')
    @patch('swak.text.read.tomllib.load')
    def test_load_called_with_defaults(self, load, mock):
        context = Mock()
        context.__enter__ = Mock(return_value='file')
        context.__exit__ = Mock()
        mock.return_value = context
        t = TomlReader(self.dir)
        _ = t('foo/bar.toml')
        load.assert_called_once_with('file', parse_float=float)

    @patch('swak.text.read.Path.open')
    @patch('swak.text.read.tomllib.load')
    def test_load_called_with_custom(self, load, mock):
        context = Mock()
        context.__enter__ = Mock(return_value='file')
        context.__exit__ = Mock()
        mock.return_value = context
        t = TomlReader(self.dir, parse_float=f)
        _ = t('foo/bar.toml')
        load.assert_called_once_with('file', parse_float=f)

    def test_read_toml_instantiation(self):
        t = TomlReader(self.dir + '/foo/bar.toml')
        actual = t()
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_toml_split(self):
        t = TomlReader(self.dir + '/foo')
        actual = t('bar.toml')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_toml_call_only(self):
        t = TomlReader()
        actual = t(self.dir + '/foo/bar.toml')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_subdir_toml_instantiation(self):
        t = TomlReader(self.dir + '/foo/hello/world.toml')
        actual = t()
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_subdir_toml_split_instantiation(self):
        t = TomlReader(self.dir + '/foo/hello/')
        actual = t('world.toml')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_subdir_toml_split_call(self):
        t = TomlReader(self.dir + '/foo/')
        actual = t('hello/world.toml')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_subdir_toml_call(self):
        t = TomlReader(self.dir)
        actual = t('foo/hello/world.toml')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_subdir_toml_call_only(self):
        t = TomlReader()
        actual = t(self.dir + '/foo/hello/world.toml')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_empty_toml(self):
        t = TomlReader(self.dir)
        actual = t('foo/empty.toml')
        self.assertDictEqual({}, actual)

    def test_path_like(self):
        t = TomlReader(self.dir)
        actual = t(Path('foo/bar.toml/'))
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_path_stripped(self):
        t = TomlReader(self.dir)
        actual = t(' foo/bar.toml/ ')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_default_raises_file_not_found(self):
        t = TomlReader(self.dir)
        with self.assertRaises(FileNotFoundError):
            _ = t('non-existing')

    def test_raises_file_not_found(self):
        t = TomlReader(self.dir, NotFound.RAISE)
        with self.assertRaises(FileNotFoundError):
            _ = t('non-existing')

    def test_warn_file_not_found(self):
        t = TomlReader(self.dir, NotFound.WARN)
        with self.assertWarns(UserWarning):
            actual = t('non-existing')
        self.assertDictEqual({}, actual)

    def test_ignore_file_not_found(self):
        t = TomlReader(self.dir, NotFound.IGNORE)
        actual = t('non-existing')
        self.assertDictEqual({}, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        t = TomlReader('hello')
        self.assertEqual("TomlReader('hello', 'raise', float)", repr(t))

    def test_custom_repr(self):
        t = TomlReader('hello', NotFound.IGNORE, f)
        self.assertEqual("TomlReader('hello', 'ignore', f)", repr(t))

    def test_pickle_works(self):
        t = TomlReader()
        _ = pickle.dumps(t)


if __name__ == '__main__':
    unittest.main()
