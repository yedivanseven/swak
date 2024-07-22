from pathlib import Path
import unittest
from unittest.mock import patch, Mock
from swak.text import TomlReader, NotFound


def f(x: str) -> float:
    return float(x)


class TestAttributes(unittest.TestCase):

    def test_dir(self):
        t = TomlReader('/hello')
        self.assertTrue(hasattr(t, 'base_dir'))
        self.assertIsInstance(t.base_dir, str)
        self.assertEqual('/hello', t.base_dir)

    def test_dir_stripped(self):
        t = TomlReader('/ hello/ ')
        self.assertTrue(hasattr(t, 'base_dir'))
        self.assertIsInstance(t.base_dir, str)
        self.assertEqual('/hello', t.base_dir)

    def test_dir_completed(self):
        t = TomlReader('hello')
        self.assertTrue(hasattr(t, 'base_dir'))
        self.assertIsInstance(t.base_dir, str)
        self.assertEqual('/hello', t.base_dir)

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

    @patch('builtins.open')
    def test_open_called(self, mock):
        mock.return_value = (self.path / 'foo/bar.toml').open('rb')
        t = TomlReader(self.dir)
        _ = t('foo/bar.toml')
        mock.assert_called_once()

    @patch('builtins.open')
    def test_open_called_with_defaults(self, mock):
        path = self.path / 'foo/bar.toml'
        mock.return_value = path.open('rb')
        t = TomlReader(self.dir)
        _ = t('foo/bar.toml')
        mock.assert_called_once_with(str(path), 'rb')

    @patch('builtins.open')
    def test_open_called_with_kwargs(self, mock):
        path = self.path / 'foo/bar.toml'
        mock.return_value = path.open('rb')
        t = TomlReader(self.dir, encoding='utf-8')
        _ = t('foo/bar.toml')
        mock.assert_called_once_with(str(path), 'rb', encoding='utf-8')

    @patch('builtins.open')
    def test_open_called_with_mode_purged(self, mock):
        path = self.path / 'foo/bar.toml'
        mock.return_value = path.open('rb')
        t = TomlReader(self.dir, encoding='utf-8', mode='w+')
        _ = t('foo/bar.toml')
        mock.assert_called_once_with(str(path), 'rb', encoding='utf-8')

    @patch('swak.text.read.tomllib.load')
    def test_load_called(self, mock):
        t = TomlReader(self.dir)
        _ = t('foo/bar.toml')
        mock.assert_called_once()

    @patch('builtins.open')
    @patch('swak.text.read.tomllib.load')
    def test_load_called_with_defaults(self, load, mock):
        context = Mock()
        context.__enter__ = Mock(return_value='file')
        context.__exit__ = Mock()
        mock.return_value = context
        t = TomlReader(self.dir)
        _ = t('foo/bar.toml')
        load.assert_called_once_with('file', parse_float=float)

    @patch('builtins.open')
    @patch('swak.text.read.tomllib.load')
    def test_load_called_with_custom(self, load, mock):
        context = Mock()
        context.__enter__ = Mock(return_value='file')
        context.__exit__ = Mock()
        mock.return_value = context
        t = TomlReader(self.dir, parse_float=f)
        _ = t('foo/bar.toml')
        load.assert_called_once_with('file', parse_float=f)

    def test_read_toml(self):
        t = TomlReader(self.dir)
        actual = t('foo/bar.toml')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_path_stripped(self):
        t = TomlReader(self.dir)
        actual = t('/ foo/bar.toml/ ')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_path_interpolated(self):
        t = TomlReader(self.dir)
        actual = t('foo/{}/world.toml', 'hello')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

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
        self.assertEqual("TomlReader('/hello', 'raise', float)", repr(t))

    def test_custom_repr(self):
        t = TomlReader('hello', NotFound.IGNORE, f)
        self.assertEqual("TomlReader('/hello', 'ignore', f)", repr(t))


if __name__ == '__main__':
    unittest.main()
