from pathlib import Path
import unittest
from unittest.mock import patch, Mock
from swak.text import YamlReader
from yaml import SafeLoader, Loader


class TestAttributes(unittest.TestCase):

    def test_dir(self):
        y = YamlReader('/hello')
        self.assertTrue(hasattr(y, 'base_dir'))
        self.assertIsInstance(y.base_dir, str)
        self.assertEqual('/hello', y.base_dir)

    def test_dir_stripped(self):
        y = YamlReader('/ hello/ ')
        self.assertTrue(hasattr(y, 'base_dir'))
        self.assertIsInstance(y.base_dir, str)
        self.assertEqual('/hello', y.base_dir)

    def test_dir_completed(self):
        y = YamlReader('hello')
        self.assertTrue(hasattr(y, 'base_dir'))
        self.assertIsInstance(y.base_dir, str)
        self.assertEqual('/hello', y.base_dir)

    def test_default_loader(self):
        y = YamlReader('hello')
        self.assertTrue(hasattr(y, 'loader'))
        self.assertIs(y.loader, Loader)

    def test_custom_loader(self):
        y = YamlReader('hello', SafeLoader)
        self.assertTrue(hasattr(y, 'loader'))
        self.assertIs(y.loader, SafeLoader)

    def test_default_kwargs(self):
        y = YamlReader('hello')
        self.assertTrue(hasattr(y, 'kwargs'))
        self.assertDictEqual({}, y.kwargs)

    def test_custom_kwargs(self):
        y = YamlReader('hello', world=42)
        self.assertTrue(hasattr(y, 'kwargs'))
        self.assertDictEqual({'world': 42}, y.kwargs)

    def test_mode_kwarg_purged(self):
        y = YamlReader('hello', world=42, mode='w+')
        self.assertTrue(hasattr(y, 'kwargs'))
        self.assertDictEqual({'world': 42}, y.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.path = Path(__file__).parent
        self.dir = str(self.path)

    def test_callable(self):
        y = YamlReader('')
        self.assertTrue(callable(y))

    @patch('builtins.open')
    def test_open_called(self, mock):
        mock.return_value = (self.path / 'foo/bar.yml').open('rb')
        y = YamlReader(self.dir)
        _ = y('foo/bar.yml')
        mock.assert_called_once()

    @patch('builtins.open')
    def test_open_called_with_defaults(self, mock):
        path = self.path / 'foo/bar.yml'
        mock.return_value = path.open('rb')
        y = YamlReader(self.dir)
        _ = y('foo/bar.yml')
        mock.assert_called_once_with(str(path), 'rb')

    @patch('builtins.open')
    def test_open_called_with_kwargs(self, mock):
        path = self.path / 'foo/bar.yml'
        mock.return_value = path.open('rb')
        y = YamlReader(self.dir, encoding='utf-8')
        _ = y('foo/bar.yml')
        mock.assert_called_once_with(str(path), 'rb', encoding='utf-8')

    @patch('builtins.open')
    def test_open_called_with_mode_purged(self, mock):
        path = self.path / 'foo/bar.yml'
        mock.return_value = path.open('rb')
        y = YamlReader(self.dir, encoding='utf-8', mode='w+')
        _ = y('foo/bar.yml')
        mock.assert_called_once_with(str(path), 'rb', encoding='utf-8')

    @patch('swak.text.read.yaml.load')
    def test_load_called(self, mock):
        y = YamlReader(self.dir)
        _ = y('foo/bar.yml')
        mock.assert_called_once()

    @patch('builtins.open')
    @patch('swak.text.read.yaml.load')
    def test_load_called_with_defaults(self, load, mock):
        context = Mock()
        context.__enter__ = Mock(return_value='file')
        context.__exit__ = Mock()
        mock.return_value = context
        y = YamlReader(self.dir)
        _ = y('foo/bar.yml')
        load.assert_called_once_with('file', Loader)

    @patch('builtins.open')
    @patch('swak.text.read.yaml.load')
    def test_load_called_with_custom(self, load, mock):
        context = Mock()
        context.__enter__ = Mock(return_value='file')
        context.__exit__ = Mock()
        mock.return_value = context
        y = YamlReader(self.dir, SafeLoader)
        _ = y('foo/bar.yml')
        load.assert_called_once_with('file', SafeLoader)

    def test_read_yaml(self):
        y = YamlReader(self.dir)
        actual = y('foo/bar.yml')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_path_stripped(self):
        y = YamlReader(self.dir)
        actual = y('/ foo/bar.yml/ ')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_path_interpolated(self):
        y = YamlReader(self.dir)
        actual = y('foo/{}/world.yml', 'hello')
        self.assertDictEqual({'world': {'answer': 42}}, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        y = YamlReader('hello')
        self.assertEqual("YamlReader('/hello', Loader)", repr(y))

    def test_custom_repr(self):
        y = YamlReader('hello', SafeLoader)
        self.assertEqual("YamlReader('/hello', SafeLoader)", repr(y))


if __name__ == '__main__':
    unittest.main()
