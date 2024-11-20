import pickle
import unittest
from unittest.mock import patch, Mock, mock_open
from pathlib import Path
from swak.text import YamlReader, NotFound
from yaml import SafeLoader, Loader


class TestAttributes(unittest.TestCase):

    def test_empty(self):
        y = YamlReader()
        self.assertTrue(hasattr(y, 'path'))
        self.assertEqual('', y.path)

    def test_path(self):
        y = YamlReader('/hello')
        self.assertTrue(hasattr(y, 'path'))
        self.assertEqual('/hello', y.path)

    def test_path_cast(self):
        y = YamlReader(123)
        self.assertEqual('123', y.path)

    def test_path_like(self):
        y = YamlReader(Path('123'))
        self.assertEqual('123', y.path)

    def test_path_stripped(self):
        y = YamlReader(' hello ')
        self.assertEqual('hello', y.path)

    def test_default_not_found(self):
        y = YamlReader('hello')
        self.assertTrue(hasattr(y, 'not_found'))
        self.assertEqual(y.not_found, NotFound.RAISE)

    def test_custom_not_found(self):
        y = YamlReader('hello', NotFound.WARN)
        self.assertTrue(hasattr(y, 'not_found'))
        self.assertEqual(y.not_found, NotFound.WARN)

    def test_default_loader(self):
        y = YamlReader('hello')
        self.assertTrue(hasattr(y, 'loader'))
        self.assertIs(y.loader, Loader)

    def test_custom_loader(self):
        y = YamlReader('hello', loader=SafeLoader)
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

    def test_open_called(self):
        mock = mock_open(read_data=b'')
        with patch('swak.text.read.Path.open', mock):
            y = YamlReader(self.dir)
            _ = y('foo/bar.yml')
        mock.assert_called_once()

    def test_open_called_with_defaults(self):
        mock = mock_open(read_data=b'')
        with patch('swak.text.read.Path.open', mock):
            y = YamlReader(self.dir)
            _ = y('foo/bar.yml')
        mock.assert_called_once_with('rb')

    def test_open_called_with_kwargs(self):
        mock = mock_open(read_data=b'')
        with patch('swak.text.read.Path.open', mock):
            y = YamlReader(self.dir, encoding='utf-8')
            _ = y('foo/bar.yml')
        mock.assert_called_once_with('rb', encoding='utf-8')

    def test_open_called_with_mode_purged(self):
        mock = mock_open(read_data=b'')
        with patch('swak.text.read.Path.open', mock):
            y = YamlReader(self.dir, encoding='utf-8', mode='w+')
            _ = y('foo/bar.yml')
        mock.assert_called_once_with('rb', encoding='utf-8')

    @patch('swak.text.read.yaml.load')
    def test_load_called(self, mock):
        y = YamlReader(self.dir)
        _ = y('foo/bar.yml')
        mock.assert_called_once()

    @patch('swak.text.read.Path.open')
    @patch('swak.text.read.yaml.load')
    def test_load_called_with_defaults(self, load, mock):
        context = Mock()
        context.__enter__ = Mock(return_value='file')
        context.__exit__ = Mock()
        mock.return_value = context
        y = YamlReader(self.dir)
        _ = y('foo/bar.yml')
        load.assert_called_once_with('file', Loader)

    @patch('swak.text.read.Path.open')
    @patch('swak.text.read.yaml.load')
    def test_load_called_with_custom(self, load, mock):
        context = Mock()
        context.__enter__ = Mock(return_value='file')
        context.__exit__ = Mock()
        mock.return_value = context
        y = YamlReader(self.dir, loader=SafeLoader)
        _ = y('foo/bar.yml')
        load.assert_called_once_with('file', SafeLoader)

    def test_read_yaml_instantiation(self):
        y = YamlReader(self.dir + '/foo/bar.yml')
        actual = y()
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_yaml_split(self):
        y = YamlReader(self.dir + '/foo/')
        actual = y(' bar.yml')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_yaml_call(self):
        y = YamlReader(self.dir)
        actual = y('foo/bar.yml')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_yaml_call_only(self):
        y = YamlReader()
        actual = y(self.dir + '/foo/bar.yml')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_subdir_yaml_instantiation(self):
        y = YamlReader(self.dir + '/foo/hello/world.yml')
        actual = y()
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_subdir_yaml_split_instantiation(self):
        y = YamlReader(self.dir + '/foo/hello ')
        actual = y(' world.yml')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_subdir_yaml_split_call(self):
        y = YamlReader(self.dir + '/foo/ ')
        actual = y('hello/world.yml')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_subdir_yaml_call(self):
        y = YamlReader(self.dir)
        actual = y('foo/hello/world.yml')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_subdir_yaml_call_only(self):
        y = YamlReader()
        actual = y(self.dir + '/foo/hello/world.yml')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_empty_yaml(self):
        y = YamlReader(self.dir)
        actual = y('foo/empty.yml')
        self.assertDictEqual({}, actual)

    def test_path_like(self):
        y = YamlReader(self.dir)
        actual = y(Path('foo/bar.yml/'))
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_path_stripped(self):
        y = YamlReader(self.dir)
        actual = y(' foo/bar.yml/ ')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_default_raises_file_not_found(self):
        y = YamlReader(self.dir)
        with self.assertRaises(FileNotFoundError):
            _ = y('non-existing')

    def test_raises_file_not_found(self):
        y = YamlReader(self.dir, NotFound.RAISE)
        with self.assertRaises(FileNotFoundError):
            _ = y('non-existing')

    def test_warn_file_not_found(self):
        y = YamlReader(self.dir, NotFound.WARN)
        with self.assertWarns(UserWarning):
            actual = y('non-existing')
        self.assertDictEqual({}, actual)

    def test_ignore_file_not_found(self):
        y = YamlReader(self.dir, NotFound.IGNORE)
        actual = y('non-existing')
        self.assertDictEqual({}, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        y = YamlReader('hello')
        self.assertEqual("YamlReader('hello', 'raise', Loader)", repr(y))

    def test_custom_repr(self):
        y = YamlReader('hello', NotFound.WARN, SafeLoader)
        self.assertEqual("YamlReader('hello', 'warn', SafeLoader)", repr(y))

    def test_pickle_works(self):
        y = YamlReader()
        _ = pickle.dumps(y)


if __name__ == '__main__':
    unittest.main()
