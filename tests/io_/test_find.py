import pickle
import unittest
from unittest.mock import patch
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.implementations.local import LocalFileSystem
from swak.io import Find, Storage


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.find = Find()

    def test_has_path(self):
        self.assertTrue(hasattr(self.find, 'path'))

    def test_path(self):
        self.assertEqual('/', self.find.path)

    def test_has_storage(self):
        self.assertTrue(hasattr(self.find, 'storage'))

    def test_storage(self):
        self.assertEqual('file', self.find.storage)

    def test_has_suffix(self):
        self.assertTrue(hasattr(self.find, 'suffix'))

    def test_suffix(self):
        self.assertEqual('', self.find.suffix)

    def test_has_max_depth(self):
        self.assertTrue(hasattr(self.find, 'max_depth'))

    def test_max_depth(self):
        self.assertIsInstance(self.find.max_depth, int)
        self.assertEqual(1, self.find.max_depth)

    def test_has_fs(self):
        self.assertTrue(hasattr(self.find, 'fs'))

    def test_fs(self):
        self.assertIsInstance(self.find.fs, LocalFileSystem)

    def test_has_prefix(self):
        self.assertTrue(hasattr(self.find, 'prefix'))

    def test_prefix(self):
        self.assertEqual('file:/', self.find.prefix)

    def test_has_non_root(self):
        self.assertTrue(hasattr(self.find, '_non_root'))

    def test_non_root(self):
        self.assertTrue(callable(self.find._non_root))

    def test_has_strip_storage(self):
        self.assertTrue(hasattr(self.find, '_STRIP_STORAGES'))
        self.assertTrue(hasattr(Find, '_STRIP_STORAGES'))

    def test_has_strip(self):
        self.assertTrue(hasattr(self.find, 'strip'))

    def test_strip(self):
        self.assertIsInstance(self.find.strip, bool)
        self.assertFalse(self.find.strip)


class TestAttributes(unittest.TestCase):

    def test_empty_path(self):
        find = Find('')
        self.assertEqual('/', find.path)

    def test_root_path(self):
        find = Find('/')
        self.assertEqual('/', find.path)

    def test_path(self):
        find = Find('/path/to/another/file.txt')
        self.assertEqual('/path/to/another/file.txt', find.path)

    def test_path_stripped(self):
        find = Find(' / path/ ')
        self.assertEqual('/path', find.path)

    def test_path_prepended(self):
        find = Find('path / ')
        self.assertEqual('/path', find.path)

    def test_path_raises(self):
        with self.assertRaises(TypeError):
            _ = Find(1)

    def test_storage_enum(self):
        find = Find('', Storage.MEMORY)
        self.assertEqual('memory', find.storage)

    def test_storage_string(self):
        find = Find('', 'memory')
        self.assertEqual('memory', find.storage)

    def test_storage_raises(self):
        with self.assertRaises(ValueError):
            _ = Find('', 4)

    def test_prefix(self):
        find = Find('', 'memory')
        self.assertEqual('memory:/', find.prefix)

    def test_fs(self):
        find = Find('', 'memory')
        self.assertIsInstance(find.fs, MemoryFileSystem)

    def test_suffix(self):
        find = Find(suffix='.gz')
        self.assertEqual('.gz', find.suffix)

    def test_suffix_strips(self):
        find = Find(suffix='. gz . ')
        self.assertEqual('.gz', find.suffix)

    def test_suffix_prepends(self):
        find = Find(suffix=' gz ')
        self.assertEqual('.gz', find.suffix)

    def test_max_depth(self):
        find = Find(max_depth=3)
        self.assertEqual(3, find.max_depth)

    def test_max_none(self):
        find = Find(max_depth=None)
        self.assertIsNone(find.max_depth)

    def test_max_depth_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            _ =  Find('', max_depth='foo')

    def test_max_depth_wrong_value_raises(self):
        with self.assertRaises(ValueError):
            _ =  Find('', max_depth=0)

    def test_storage_kws(self):
        find = Find(storage_kws={'answer': 42})
        self.assertDictEqual({'answer': 42}, find.storage_kws)

    def test_strip(self):
        find = Find(storage=Storage.GCS)
        self.assertTrue(find.strip)


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'

    def test_non_root_empty(self):
        find = Find(self.path)
        path = find._non_root()
        self.assertIsInstance(path, str)
        self.assertEqual(self.path, path)

    def test_non_root_empty_string(self):
        find = Find(self.path)
        path = find._non_root('')
        self.assertIsInstance(path, str)
        self.assertEqual(self.path, path)

    def test_non_root_appends(self):
        find = Find('/path/to/')
        path = find._non_root('sub/file.txt')
        self.assertEqual('/path/to/sub/file.txt', path)

    def test_non_root_strips(self):
        find = Find('/path/to/')
        path = find._non_root(' sub/file.txt / ')
        self.assertEqual('/path/to/sub/file.txt', path)

    def test_non_root_replaces(self):
        find = Find('/path/to/file.txt')
        path = find._non_root('/another/different.txt')
        self.assertEqual('/another/different.txt', path)

    def test_non_root_raises_on_root(self):
        find = Find('/')
        with self.assertRaises(ValueError):
            _ = find._non_root('/')

    def test_root_strips_funky_file_systems_init(self):
        for storage in Find._STRIP_STORAGES:
            find = Find('/path/to', storage)
            actual = find._non_root()
            self.assertFalse(actual.startswith('/'))

    def test_root_strips_funky_file_systems_call(self):
        for storage in Find._STRIP_STORAGES:
            find = Find('/path/to', storage)
            actual = find._non_root('subdir/file.txt')
            self.assertFalse(actual.startswith('/'))

    def test_root_strips_funky_file_systems_reset(self):
        for storage in Find._STRIP_STORAGES:
            find = Find('/path/to', storage)
            actual = find._non_root('/different/path')
            self.assertFalse(actual.startswith('/'))


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'
        self.storage = Storage.MEMORY

    def test_callable(self):
        find = Find()
        self.assertTrue(callable(find))

    def test_non_root_called_empty(self):
        find = Find(self.path, storage=self.storage)
        with patch.object(find, '_non_root') as non_root:
            non_root.return_value = self.path
            _ = find()
            non_root.assert_called_once_with('')

    def test_non_root_called_empty_string(self):
        find = Find(self.path, storage=self.storage)
        with patch.object(find, '_non_root') as non_root:
            non_root.return_value = self.path
            _ = find('')
            non_root.assert_called_once_with('')

    def test_non_root_called_string(self):
        find = Find(self.path, storage=self.storage)
        with patch.object(find, '_non_root') as non_root:
            non_root.return_value = self.path
            _ = find('string')
            non_root.assert_called_once_with('string')

    def test_find_called_defaults(self):
        find = Find(self.path, storage=self.storage)
        with patch.object(
                find,
                '_non_root'
        ) as non_root, patch.object(
            find.fs,
            'find'
        ) as discover:
            non_root.return_value = '/different/path'
            _ = find()
            discover.assert_called_once_with(
                '/different/path',
                maxdepth=find.max_depth,
                withdirs=False,
                detail=False
            )

    def test_find_called_custom(self):
        find = Find(self.path, storage=self.storage, max_depth=12)
        with patch.object(
                find,
                '_non_root'
        ) as non_root, patch.object(
            find.fs,
            'find'
        ) as discover:
            non_root.return_value = '/different/path'
            _ = find()
            discover.assert_called_once_with(
                '/different/path',
                maxdepth=12,
                withdirs=False,
                detail=False
            )

    def test_prefix_removed(self):
        find = Find(self.path, storage=self.storage)
        with patch.object(
            find.fs,
            'find'
        ) as discover:
            discover.return_value = [
                'memory://foo/bar.txt',
                '/hello/world.txt'
            ]
            actual = find()
        expected = ['/foo/bar.txt', '/hello/world.txt']
        self.assertListEqual(expected, actual)

    def test_prefix_not_removed(self):
        expected = ['s3://foo/bar.txt', 'gcs://hello/world.txt']
        find = Find(self.path, storage=self.storage)
        with patch.object(
            find.fs,
            'find'
        ) as find:
            find.return_value = expected
            actual = find()
        self.assertListEqual(expected, actual)

    def test_suffix_filtered(self):
        find = Find(self.path, storage=self.storage, suffix='.gz')
        with patch.object(
            find.fs,
            'find'
        ) as discover:
            discover.return_value = [
                '/path/file.txt',
                '/foo/bar.gz',
                '/hello/world'
            ]
            actual = find()
        self.assertListEqual(['/foo/bar.gz'], actual)

    def test_suffix_allowed(self):
        find = Find(self.path, storage=self.storage)
        expected = [
                '/path/file.txt',
                '/foo/bar.gz',
                '/hello/world'
            ]
        with patch.object(
            find.fs,
            'find'
        ) as discover:
            discover.return_value = expected
            actual = find()
        self.assertListEqual(expected, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        find = Find()
        expected = "Find('/', 'file', '', 1, {})"
        self.assertEqual(expected, repr(find))

    def test_custom_repr(self):
        find = Find('/path/to', 'memory', '.gz', 3, {'answer': 42})
        expected = "Find('/path/to', 'memory', '.gz', 3, {'answer': 42})"
        self.assertEqual(expected, repr(find))

    def test_pickle_works(self):
        find = Find('/path/to', 'memory', '.gz', 3, {'answer': 42})
        _ = pickle.loads(pickle.dumps(find))


if __name__ == '__main__':
    unittest.main()
