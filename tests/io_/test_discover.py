import pickle
import unittest
from unittest.mock import patch
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.implementations.local import LocalFileSystem
from swak.io import Discover, Storage


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.discover = Discover()

    def test_has_path(self):
        self.assertTrue(hasattr(self.discover, 'path'))

    def test_path(self):
        self.assertEqual('/', self.discover.path)

    def test_has_storage(self):
        self.assertTrue(hasattr(self.discover, 'storage'))

    def test_storage(self):
        self.assertEqual('file', self.discover.storage)

    def test_has_suffix(self):
        self.assertTrue(hasattr(self.discover, 'suffix'))

    def test_suffix(self):
        self.assertEqual('', self.discover.suffix)

    def test_has_max_depth(self):
        self.assertTrue(hasattr(self.discover, 'max_depth'))

    def test_max_depth(self):
        self.assertIsInstance(self.discover.max_depth, int)
        self.assertEqual(1, self.discover.max_depth)

    def test_has_fs(self):
        self.assertTrue(hasattr(self.discover, 'fs'))

    def test_fs(self):
        self.assertIsInstance(self.discover.fs, LocalFileSystem)

    def test_has_prefix(self):
        self.assertTrue(hasattr(self.discover, 'prefix'))

    def test_prefix(self):
        self.assertEqual('file:/', self.discover.prefix)

    def test_has_non_root(self):
        self.assertTrue(hasattr(self.discover, '_non_root'))

    def test_non_root(self):
        self.assertTrue(callable(self.discover._non_root))


class TestAttributes(unittest.TestCase):

    def test_empty_path(self):
        discover = Discover('')
        self.assertEqual('/', discover.path)

    def test_root_path(self):
        discover = Discover('/')
        self.assertEqual('/', discover.path)

    def test_path(self):
        discover = Discover('/path/to/another/file.txt')
        self.assertEqual('/path/to/another/file.txt', discover.path)

    def test_path_stripped(self):
        discover = Discover(' / path/ ')
        self.assertEqual('/path', discover.path)

    def test_path_prepended(self):
        discover = Discover('path / ')
        self.assertEqual('/path', discover.path)

    def test_path_raises(self):
        with self.assertRaises(TypeError):
            _ = Discover(1)

    def test_storage_enum(self):
        discover = Discover('', Storage.MEMORY)
        self.assertEqual('memory', discover.storage)

    def test_storage_string(self):
        discover = Discover('', 'memory')
        self.assertEqual('memory', discover.storage)

    def test_storage_raises(self):
        with self.assertRaises(ValueError):
            _ = Discover('', 4)

    def test_prefix(self):
        discover = Discover('', 'memory')
        self.assertEqual('memory:/', discover.prefix)

    def test_fs(self):
        discover = Discover('', 'memory')
        self.assertIsInstance(discover.fs, MemoryFileSystem)

    def test_suffix(self):
        discover = Discover(suffix='.gz')
        self.assertEqual('.gz', discover.suffix)

    def test_suffix_strips(self):
        discover = Discover(suffix='. gz . ')
        self.assertEqual('.gz', discover.suffix)

    def test_suffix_prepends(self):
        discover = Discover(suffix=' gz ')
        self.assertEqual('.gz', discover.suffix)

    def test_max_depth(self):
        discover = Discover(max_depth=3)
        self.assertEqual(3, discover.max_depth)

    def test_max_none(self):
        discover = Discover(max_depth=None)
        self.assertIsNone(discover.max_depth)

    def test_max_depth_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            _ =  Discover('', max_depth='foo')

    def test_max_depth_wrong_value_raises(self):
        with self.assertRaises(ValueError):
            _ =  Discover('', max_depth=0)

    def test_storage_kws(self):
        write = Discover(storage_kws={'answer': 42})
        self.assertDictEqual({'answer': 42}, write.storage_kws)


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'

    def test_non_root_empty(self):
        discover = Discover(self.path)
        path = discover._non_root()
        self.assertIsInstance(path, str)
        self.assertEqual(self.path, path)

    def test_non_root_empty_string(self):
        discover = Discover(self.path)
        path = discover._non_root('')
        self.assertIsInstance(path, str)
        self.assertEqual(self.path, path)

    def test_non_root_appends(self):
        discover = Discover('/path/to/')
        path = discover._non_root('sub/file.txt')
        self.assertEqual('/path/to/sub/file.txt', path)

    def test_non_root_strips(self):
        discover = Discover('/path/to/')
        path = discover._non_root(' sub/file.txt / ')
        self.assertEqual('/path/to/sub/file.txt', path)

    def test_non_root_replaces(self):
        discover = Discover('/path/to/file.txt')
        path = discover._non_root('/another/different.txt')
        self.assertEqual('/another/different.txt', path)

    def test_non_root_raises_on_root(self):
        discover = Discover('/')
        with self.assertRaises(ValueError):
            _ = discover._non_root('/')


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'
        self.storage = Storage.MEMORY

    def test_callable(self):
        discover = Discover()
        self.assertTrue(callable(discover))

    def test_non_root_called_empty(self):
        discover = Discover(self.path, storage=self.storage)
        with patch.object(discover, '_non_root') as non_root:
            non_root.return_value = self.path
            _ = discover()
            non_root.assert_called_once_with('')

    def test_non_root_called_empty_string(self):
        discover = Discover(self.path, storage=self.storage)
        with patch.object(discover, '_non_root') as non_root:
            non_root.return_value = self.path
            _ = discover('')
            non_root.assert_called_once_with('')

    def test_non_root_called_string(self):
        discover = Discover(self.path, storage=self.storage)
        with patch.object(discover, '_non_root') as non_root:
            non_root.return_value = self.path
            _ = discover('string')
            non_root.assert_called_once_with('string')

    def test_find_called_defaults(self):
        discover = Discover(self.path, storage=self.storage)
        with patch.object(
                discover,
                '_non_root'
        ) as non_root, patch.object(
            discover.fs,
            'find'
        ) as find:
            non_root.return_value = '/different/path'
            _ = discover()
            find.assert_called_once_with(
                '/different/path',
                maxdepth=discover.max_depth,
                withdirs=False,
                detail=False
            )

    def test_find_called_custom(self):
        discover = Discover(self.path, storage=self.storage, max_depth=12)
        with patch.object(
                discover,
                '_non_root'
        ) as non_root, patch.object(
            discover.fs,
            'find'
        ) as find:
            non_root.return_value = '/different/path'
            _ = discover()
            find.assert_called_once_with(
                '/different/path',
                maxdepth=12,
                withdirs=False,
                detail=False
            )

    def test_prefix_removed(self):
        discover = Discover(self.path, storage=self.storage)
        with patch.object(
            discover.fs,
            'find'
        ) as find:
            find.return_value = ['memory://foo/bar.txt', '/hello/world.txt']
            actual = discover()
        expected = ['/foo/bar.txt', '/hello/world.txt']
        self.assertListEqual(expected, actual)

    def test_prefix_not_removed(self):
        expected = ['s3://foo/bar.txt', 'gcs://hello/world.txt']
        discover = Discover(self.path, storage=self.storage)
        with patch.object(
            discover.fs,
            'find'
        ) as find:
            find.return_value = expected
            actual = discover()
        self.assertListEqual(expected, actual)

    def test_suffix_filtered(self):
        discover = Discover(self.path, storage=self.storage, suffix='.gz')
        with patch.object(
            discover.fs,
            'find'
        ) as find:
            find.return_value = [
                '/path/file.txt',
                '/foo/bar.gz',
                '/hello/world'
            ]
            actual = discover()
        self.assertListEqual(['/foo/bar.gz'], actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        discover = Discover()
        expected = "Discover('/', 'file', '', 1, {})"
        self.assertEqual(expected, repr(discover))

    def test_custom_repr(self):
        discover = Discover('/path/to', 'memory', '.gz', 3, {'answer': 42})
        expected = "Discover('/path/to', 'memory', '.gz', 3, {'answer': 42})"
        self.assertEqual(expected, repr(discover))

    def test_pickle_works(self):
        discover = Discover('/path/to', 'memory', '.gz', 3, {'answer': 42})
        _ = pickle.loads(pickle.dumps(discover))


if __name__ == '__main__':
    unittest.main()
