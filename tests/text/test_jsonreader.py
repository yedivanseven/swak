import pickle
import unittest
from unittest.mock import patch, Mock, mock_open
from pathlib import Path
from swak.misc import NotFound
from swak.text import JsonReader
from json import JSONDecodeError


class TestAttributes(unittest.TestCase):

    def test_emptj(self):
        j = JsonReader()
        self.assertTrue(hasattr(j, 'path'))
        self.assertEqual('', j.path)

    def test_path(self):
        j = JsonReader('/hello')
        self.assertTrue(hasattr(j, 'path'))
        self.assertEqual('/hello', j.path)

    def test_path_cast(self):
        j = JsonReader(123)
        self.assertEqual('123', j.path)

    def test_path_like(self):
        j = JsonReader(Path('123'))
        self.assertEqual('123', j.path)

    def test_path_stripped(self):
        j = JsonReader(' hello ')
        self.assertEqual('hello', j.path)

    def test_default_not_found(self):
        j = JsonReader('hello')
        self.assertTrue(hasattr(j, 'not_found'))
        self.assertEqual(j.not_found, NotFound.RAISE)

    def test_custom_not_found(self):
        j = JsonReader('hello', NotFound.WARN)
        self.assertTrue(hasattr(j, 'not_found'))
        self.assertEqual(j.not_found, NotFound.WARN)

    def test_default_gzipped(self):
        j = JsonReader('hello')
        self.assertTrue(hasattr(j, 'gzipped'))
        self.assertIsNone(j.gzipped)

    def test_custom_gzipped(self):
        obj = object()
        j = JsonReader('hello', gzipped=obj)
        self.assertIs(j.gzipped, obj)

    def test_default_kwargs(self):
        j = JsonReader('hello')
        self.assertTrue(hasattr(j, 'kwargs'))
        self.assertDictEqual({}, j.kwargs)

    def test_custom_kwargs(self):
        j = JsonReader('hello', world=42)
        self.assertTrue(hasattr(j, 'kwargs'))
        self.assertDictEqual({'world': 42}, j.kwargs)

    def test_mode_kwarg_purged(self):
        j = JsonReader('hello', world=42, mode='w+')
        self.assertTrue(hasattr(j, 'kwargs'))
        self.assertDictEqual({'world': 42}, j.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.path = Path(__file__).parent
        self.dir = str(self.path)

    def test_callable(self):
        j = JsonReader('')
        self.assertTrue(callable(j))

    def test_open_called(self):
        mock = mock_open(read_data='{}')
        with patch('swak.text.read.Path.open', mock):
            j = JsonReader(self.dir)
            _ = j('foo/bar.json')
        mock.assert_called_once()

    def test_open_called_with_defaults(self):
        mock = mock_open(read_data='{}')
        with patch('swak.text.read.Path.open', mock):
            j = JsonReader(self.dir)
            _ = j('foo/bar.json')
        mock.assert_called_once_with('rt')

    def test_open_called_with_kwargs(self):
        mock = mock_open(read_data='{}')
        with patch('swak.text.read.Path.open', mock):
            j = JsonReader(self.dir, encoding='utf-8')
            _ = j('foo/bar.json')
        mock.assert_called_once_with('rt', encoding='utf-8')

    def test_open_called_with_mode_purged(self):
        mock = mock_open(read_data='{}')
        with patch('swak.text.read.Path.open', mock):
            j = JsonReader(self.dir, encoding='utf-8', mode='w+')
            _ = j('foo/bar.json')
        mock.assert_called_once_with('rt', encoding='utf-8')

    def test_gzip_open_called_explicit(self):
        mock = mock_open(read_data='{}')
        with patch('swak.text.read.gzip.open', mock):
            j = JsonReader(self.dir, gzipped=True)
            _ = j('foo/bar.json')
        mock.assert_called_once()

    def test_gzip_open_called_implicit(self):
        mock = mock_open(read_data='{}')
        with patch('swak.text.read.gzip.open', mock):
            j = JsonReader(self.dir)
            _ = j('foo/bar.json.gz')
        mock.assert_called_once()

    def test_gzip_open_called_explicitly_with_defaults(self):
        mock = mock_open(read_data='{}')
        file = 'foo/bar.json'
        path = Path(self.dir) / file
        with patch('swak.text.read.gzip.open', mock):
            j = JsonReader(self.dir, gzipped=True)
            _ = j(file)
        mock.assert_called_once_with(path, 'rt')

    def test_gzip_open_called_implicitly_with_defaults(self):
        mock = mock_open(read_data='{}')
        file = 'foo/bar.json.gz'
        path = Path(self.dir) / file
        with patch('swak.text.read.gzip.open', mock):
            j = JsonReader(self.dir)
            _ = j(file)
        mock.assert_called_once_with(path, 'rt')

    def test_gzip_open_called_explicitly_with_kwargs(self):
        mock = mock_open(read_data='{}')
        file = 'foo/bar.json'
        path = Path(self.dir) / file
        with patch('swak.text.read.gzip.open', mock):
            j = JsonReader(self.dir, gzipped=True, encoding='utf-8')
            _ = j(file)
        mock.assert_called_once_with(path, 'rt', encoding='utf-8')

    def test_gzip_open_called_implicitly_with_kwargs(self):
        mock = mock_open(read_data='{}')
        file = 'foo/bar.json.gz'
        path = Path(self.dir) / file
        with patch('swak.text.read.gzip.open', mock):
            j = JsonReader(self.dir, encoding='utf-8')
            _ = j(file)
        mock.assert_called_once_with(path, 'rt', encoding='utf-8')

    def test_gzip_open_called_explicitly_with_mode_purged(self):
        mock = mock_open(read_data='{}')
        file = 'foo/bar.json'
        path = Path(self.dir) / file
        with patch('swak.text.read.gzip.open', mock):
            j = JsonReader(self.dir, gzipped=True, encoding='utf-8', mode='w+')
            _ = j(file)
        mock.assert_called_once_with(path, 'rt', encoding='utf-8')

    def test_gzip_open_called_implicitly_with_mode_purged(self):
        mock = mock_open(read_data='{}')
        file = 'foo/bar.json'
        path = Path(self.dir) / file
        with patch('swak.text.read.gzip.open', mock):
            j = JsonReader(self.dir, gzipped=True, encoding='utf-8', mode='w+')
            _ = j(file)
        mock.assert_called_once_with(path, 'rt', encoding='utf-8')

    @patch('swak.text.read.json.load')
    def test_load_called(self, mock):
        j = JsonReader(self.dir)
        _ = j('foo/bar.json')
        mock.assert_called_once()

    @patch('swak.text.read.Path.open')
    @patch('swak.text.read.json.load')
    def test_load_called_with_defaults(self, load, mock):
        context = Mock()
        context.__enter__ = Mock(return_value='file')
        context.__exit__ = Mock()
        mock.return_value = context
        j = JsonReader(self.dir)
        _ = j('foo/bar.json')
        load.assert_called_once_with('file')

    @patch('swak.text.read.gzip.open')
    @patch('swak.text.read.json.load')
    def test_load_called_with_explicit_gzip(self, load, mock):
        context = Mock()
        context.__enter__ = Mock(return_value='file')
        context.__exit__ = Mock()
        mock.return_value = context
        j = JsonReader(self.dir, gzipped=True)
        _ = j('foo/bar.json')
        load.assert_called_once_with('file')

    @patch('swak.text.read.gzip.open')
    @patch('swak.text.read.json.load')
    def test_load_called_with_implicit_gzip(self, load, mock):
        context = Mock()
        context.__enter__ = Mock(return_value='file')
        context.__exit__ = Mock()
        mock.return_value = context
        j = JsonReader(self.dir, gzipped=True)
        _ = j('foo/bar.json.gz')
        load.assert_called_once_with('file')

    def test_read_implicit_json_instantiation(self):
        j = JsonReader(self.dir + '/foo/bar.json')
        actual = j()
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_explicit_json_instantiation(self):
        j = JsonReader(self.dir + '/foo/bar.json', gzipped=False)
        actual = j()
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_implicit_json_split(self):
        j = JsonReader(self.dir + '/foo/')
        actual = j(' bar.json')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_explicit_json_split(self):
        j = JsonReader(self.dir + '/foo/', gzipped=False)
        actual = j(' bar.json')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_implicit_json_call(self):
        j = JsonReader(self.dir)
        actual = j('foo/bar.json')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_explicit_json_call(self):
        j = JsonReader(self.dir, gzipped=False)
        actual = j('foo/bar.json')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_implicit_json_call_only(self):
        j = JsonReader()
        actual = j(self.dir + '/foo/bar.json')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_explicit_json_call_only(self):
        j = JsonReader(gzipped=False)
        actual = j(self.dir + '/foo/bar.json')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_implicit_subdir_json_instantiation(self):
        j = JsonReader(self.dir + '/foo/hello/world.json')
        actual = j()
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_explicit_subdir_json_instantiation(self):
        j = JsonReader(self.dir + '/foo/hello/world.json', gzipped=False)
        actual = j()
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_implicit_subdir_json_split_instantiation(self):
        j = JsonReader(self.dir + '/foo/hello ')
        actual = j(' world.json')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_explicit_subdir_json_split_instantiation(self):
        j = JsonReader(self.dir + '/foo/hello ', gzipped=False)
        actual = j(' world.json')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_implicit_subdir_json_split_call(self):
        j = JsonReader(self.dir + '/foo/ ')
        actual = j('hello/world.json')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_explicit_subdir_json_split_call(self):
        j = JsonReader(self.dir + '/foo/ ', gzipped=False)
        actual = j('hello/world.json')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_implicit_subdir_json_call(self):
        j = JsonReader(self.dir)
        actual = j('foo/hello/world.json')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_explicit_subdir_json_call(self):
        j = JsonReader(self.dir, gzipped=False)
        actual = j('foo/hello/world.json')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_implicit_subdir_json_call_only(self):
        j = JsonReader()
        actual = j(self.dir + '/foo/hello/world.json')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_explicit_subdir_json_call_only(self):
        j = JsonReader(gzipped=False)
        actual = j(self.dir + '/foo/hello/world.json')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_empty_json_raises(self):
        j = JsonReader(self.dir)
        with self.assertRaises(JSONDecodeError):
            _ = j('foo/empty.json')

    def test_read_explicit_gzip_json_instantiation(self):
        j = JsonReader(self.dir + '/foo/bar.json.gz', gzipped=True)
        actual = j()
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_implicit_gzip_json_instantiation(self):
        j = JsonReader(self.dir + '/foo/bar.json.gz')
        actual = j()
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_explicit_gzip_json_split(self):
        j = JsonReader(self.dir + '/foo/', gzipped=True)
        actual = j(' bar.json.gz')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_implicit_gzip_json_call(self):
        j = JsonReader(self.dir)
        actual = j('foo/bar.json.gz')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_explicit_gzip_json_call(self):
        j = JsonReader(self.dir, gzipped=True)
        actual = j('foo/bar.json.gz')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_implicit_gzip_json_call_only(self):
        j = JsonReader()
        actual = j(self.dir + '/foo/bar.json.gz')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_explicit_gzip_json_call_only(self):
        j = JsonReader(gzipped=True)
        actual = j(self.dir + '/foo/bar.json.gz')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_read_implicit_subdir_gzip_json_instantiation(self):
        j = JsonReader(self.dir + '/foo/hello/world.json.gz')
        actual = j()
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_explicit_subdir_gzip_json_instantiation(self):
        j = JsonReader(self.dir + '/foo/hello/world.json.gz', gzipped=True)
        actual = j()
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_implicit_subdir_gzip_json_split_instantiation(self):
        j = JsonReader(self.dir + '/foo/hello ')
        actual = j(' world.json.gz')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_explicit_subdir_gzip_json_split_instantiation(self):
        j = JsonReader(self.dir + '/foo/hello ', gzipped=True)
        actual = j(' world.json.gz')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_implicit_subdir_gzip_json_split_call(self):
        j = JsonReader(self.dir + '/foo/ ')
        actual = j('hello/world.json.gz')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_explicit_subdir_gzip_json_split_call(self):
        j = JsonReader(self.dir + '/foo/ ', gzipped=True)
        actual = j('hello/world.json.gz')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_implicit_subdir_gzip_json_call(self):
        j = JsonReader(self.dir)
        actual = j('foo/hello/world.json.gz')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_explicit_subdir_gzip_json_call(self):
        j = JsonReader(self.dir, gzipped=True)
        actual = j('foo/hello/world.json.gz')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_implicit_subdir_gzip_json_call_only(self):
        j = JsonReader()
        actual = j(self.dir + '/foo/hello/world.json.gz')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_explicit_subdir_gzip_json_call_only(self):
        j = JsonReader(gzipped=True)
        actual = j(self.dir + '/foo/hello/world.json.gz')
        self.assertDictEqual({'world': {'answer': 42}}, actual)

    def test_read_empty_gzip_json_raises(self):
        j = JsonReader(self.dir)
        with self.assertRaises(JSONDecodeError):
            _ = j('foo/empty.json.gz')

    def test_path_like(self):
        j = JsonReader(self.dir)
        actual = j(Path('foo/bar.json/'))
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_path_stripped(self):
        j = JsonReader(self.dir)
        actual = j(' foo/bar.json/ ')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_default_raises_file_not_found(self):
        j = JsonReader(self.dir)
        with self.assertRaises(FileNotFoundError):
            _ = j('non-existing')

    def test_raises_file_not_found(self):
        j = JsonReader(self.dir, NotFound.RAISE)
        with self.assertRaises(FileNotFoundError):
            _ = j('non-existing')

    def test_warn_file_not_found(self):
        j = JsonReader(self.dir, NotFound.WARN)
        with self.assertWarns(UserWarning):
            actual = j('non-existing')
        self.assertDictEqual({}, actual)

    def test_ignore_file_not_found(self):
        j = JsonReader(self.dir, NotFound.IGNORE)
        actual = j('non-existing')
        self.assertDictEqual({}, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        j = JsonReader('hello')
        expected = "JsonReader('hello', 'raise', None)"
        self.assertEqual(expected, repr(j))

    def test_custom_repr(self):
        j = JsonReader('hello', NotFound.WARN, True, encoding='utf-8')
        expected = "JsonReader('hello', 'warn', True, encoding='utf-8')"
        self.assertEqual(expected, repr(j))

    def test_pickle_works(self):
        j = JsonReader()
        _ = pickle.loads(pickle.dumps(j))


if __name__ == '__main__':
    unittest.main()
