import pickle
import unittest
from swak.pl.io import LazyWriter, LazyStorage


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'
        self.write = LazyWriter(self.path)

    def test_has_path(self):
        self.assertTrue(hasattr(self.write, 'path'))

    def test_path(self):
        self.assertEqual(self.path, self.write.path)

    def test_has_storage(self):
        self.assertTrue(hasattr(self.write, 'storage'))

    def test_storage(self):
        self.assertEqual('file', self.write.storage)


    def test_has_storage_kws(self):
        self.assertTrue(hasattr(self.write, 'storage_kws'))

    def test_storage_kws(self):
        self.assertDictEqual({}, self.write.storage_kws)

    def test_has_prefix(self):
        self.assertTrue(hasattr(self.write, 'prefix'))

    def test_prefix(self):
        self.assertEqual('', self.write.prefix)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'
        self.storage = str(LazyStorage.AZURE)

    def test_empty_path(self):
        write = LazyWriter('', self.storage)
        self.assertEqual('/', write.path)

    def test_root_path(self):
        write = LazyWriter('/', self.storage)
        self.assertEqual('/', write.path)

    def test_path(self):
        write = LazyWriter('/path/to/another/file.txt', self.storage)
        self.assertEqual('/path/to/another/file.txt', write.path)

    def test_path_stripped(self):
        write = LazyWriter(' / path/ ', self.storage)
        self.assertEqual('/path', write.path)

    def test_path_prepended(self):
        write = LazyWriter('path / ', self.storage)
        self.assertEqual('/path', write.path)

    def test_path_raises(self):
        with self.assertRaises(TypeError):
            _ = LazyWriter(1, self.storage)

    def test_storage(self):
        write = LazyWriter(self.path, self.storage)
        self.assertEqual(self.storage, write.storage)

    def test_storage_raises(self):
        with self.assertRaises(ValueError):
            _ = LazyWriter(self.path, 4)

    def test_storage_kws(self):
        write = LazyWriter(self.path, self.storage, storage_kws={'answer': 42})
        self.assertDictEqual({'answer': 42}, write.storage_kws)

    def test_prefix(self):
        write = LazyWriter(self.path, self.storage)
        self.assertEqual(f'{self.storage}:/', write.prefix)


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'
        self.storage = str(LazyStorage.GCS)

    def test_has_uri_from(self):
        write = LazyWriter(self.path, self.storage)
        self.assertTrue(hasattr(write, '_uri_from'))
        self.assertTrue(callable(write._uri_from))

    def test_uri_from_empty(self):
        write = LazyWriter(self.path, self.storage,)
        path = write._uri_from()
        self.assertIsInstance(path, str)
        self.assertEqual('gs:/' + self.path, path)

    def test_uri_from_interpolates(self):
        write = LazyWriter('/path/to/{}.txt', self.storage)
        path = write._uri_from('file')
        self.assertEqual('gs:/' + self.path, path)

    def test_uri_from_strips(self):
        write = LazyWriter('/path/to/{}.txt {}', self.storage)
        path = write._uri_from('file', '/')
        self.assertEqual('gs:/' + self.path, path)

    def test_uri_raises_on_root(self):
        write = LazyWriter('/{}.txt', self.storage)
        with self.assertRaises(ValueError):
            _ = write._uri_from('file')


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'

    def test_default_repr(self):
        write = LazyWriter(self.path)
        expected = "LazyWriter('/path/to/file.txt', 'file', {})"
        self.assertEqual(expected, repr(write))

    def test_custom_repr(self):
        write = LazyWriter(
            self.path,
            'az',
            {'answer': 42},
            'foo',
            bar='baz'
        )
        expected = ("LazyWriter('/path/to/file.txt', 'az',"
                    " {'answer': 42}, 'foo', bar='baz')")
        self.assertEqual(expected, repr(write))

    def test_pickle_works(self):
        write = LazyWriter(self.path)
        _ = pickle.loads(pickle.dumps(write))


if __name__ == '__main__':
    unittest.main()
