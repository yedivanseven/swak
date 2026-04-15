import pickle
import unittest
from swak.pl.io import LazyStorage, LazyReader


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.read = LazyReader()

    def test_has_path(self):
        self.assertTrue(hasattr(self.read, 'path'))

    def test_path(self):
        self.assertEqual('/', self.read.path)

    def test_has_storage(self):
        self.assertTrue(hasattr(self.read, 'storage'))

    def test_storage(self):
        self.assertEqual(LazyStorage.FILE, self.read.storage)

    def test_has_storage_kws(self):
        self.assertTrue(hasattr(self.read, 'storage_kws'))

    def test_storage_kws(self):
        self.assertDictEqual({}, self.read.storage_kws)

    def test_has_prefix(self):
        self.assertTrue(hasattr(self.read, 'prefix'))

    def test_prefix(self):
        self.assertEqual('', self.read.prefix)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'
        self.storage = str(LazyStorage.AZURE)

    def test_empty_path(self):
        read = LazyReader('', self.storage)
        self.assertEqual('/', read.path)

    def test_root_path(self):
        read = LazyReader('/', self.storage)
        self.assertEqual('/', read.path)

    def test_path(self):
        read = LazyReader('/path/to/another/file.txt', self.storage)
        self.assertEqual('/path/to/another/file.txt', read.path)

    def test_path_stripped(self):
        read = LazyReader(' / path/ ', self.storage)
        self.assertEqual('/path', read.path)

    def test_path_prepended(self):
        read = LazyReader('path / ', self.storage)
        self.assertEqual('/path', read.path)

    def test_path_raises(self):
        with self.assertRaises(TypeError):
            _ = LazyReader(1, self.storage)

    def test_storage(self):
        read = LazyReader(self.path, self.storage)
        self.assertEqual(self.storage, read.storage)

    def test_storage_raises(self):
        with self.assertRaises(ValueError):
            _ = LazyReader(self.path, 4)

    def test_storage_kws(self):
        read = LazyReader(self.path, self.storage, storage_kws={'answer': 42})
        self.assertDictEqual({'answer': 42}, read.storage_kws)

    def test_prefix(self):
        read = LazyReader(self.path, self.storage)
        self.assertEqual(f'{self.storage}:/', read.prefix)


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'
        self.storage = str(LazyStorage.HF)

    def test_has_non_root(self):
        read = LazyReader(self.path, self.storage)
        self.assertTrue(hasattr(read, '_non_root'))
        self.assertTrue(callable(read._non_root))

    def test_non_root_empty(self):
        read = LazyReader(self.path, self.storage)
        path = read._non_root()
        self.assertIsInstance(path, str)
        self.assertEqual('hf:/' + self.path, path)

    def test_non_root_appends(self):
        read = LazyReader('/path/to/', self.storage)
        path = read._non_root('sub/file.txt')
        self.assertEqual('hf://path/to/sub/file.txt', path)

    def test_non_root_strips(self):
        read = LazyReader('/path/to/', self.storage)
        path = read._non_root(' sub/file.txt / ')
        self.assertEqual('hf://path/to/sub/file.txt', path)

    def test_non_root_replaces(self):
        read = LazyReader('/path/to/file.txt', self.storage)
        path = read._non_root('/another/different.txt')
        self.assertEqual('hf://another/different.txt', path)

    def test_non_root_raises_on_root(self):
        read = LazyReader('/', self.storage)
        with self.assertRaises(ValueError):
            _ = read._non_root('file.txt')


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'

    def test_default_repr(self):
        read = LazyReader(self.path)
        expected = "LazyReader('/path/to/file.txt', 'file', {})"
        self.assertEqual(expected, repr(read))

    def test_custom_repr(self):
        read = LazyReader(
            self.path,
            'az',
            {'answer': 42},
            'foo',
            bar='baz'
        )
        expected = ("LazyReader('/path/to/file.txt', 'az',"
                    " {'answer': 42}, 'foo', bar='baz')")
        self.assertEqual(expected, repr(read))

    def test_pickle_works(self):
        read = LazyReader(self.path)
        _ = pickle.loads(pickle.dumps(read))


if __name__ == '__main__':
    unittest.main()
