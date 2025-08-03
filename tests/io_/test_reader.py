import pickle
import unittest
from unittest.mock import Mock, patch, mock_open
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.implementations.local import LocalFileSystem
from swak.io import Reader, Storage, Mode, Compression


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'
        self.storage = str(Storage.MEMORY)
        self.read = Reader(self.path, Storage.MEMORY)

    def test_has_path(self):
        self.assertTrue(hasattr(self.read, 'path'))

    def test_path(self):
        self.assertEqual(self.path, self.read.path)

    def test_has_storage(self):
        self.assertTrue(hasattr(self.read, 'storage'))

    def test_storage(self):
        self.assertEqual(self.storage, self.read.storage)

    def test_has_mode(self):
        self.assertTrue(hasattr(self.read, 'mode'))

    def test_mode(self):
        self.assertEqual('rb', self.read.mode)

    def test_has_chunk_size(self):
        self.assertTrue(hasattr(self.read, 'chunk_size'))

    def test_chunk_size(self):
        self.assertIsInstance(self.read.chunk_size, float)
        self.assertEqual(32.0, self.read.chunk_size)

    def test_has_storage_kws(self):
        self.assertTrue(hasattr(self.read, 'storage_kws'))

    def test_storage_kws(self):
        self.assertDictEqual({}, self.read.storage_kws)

    def test_has_chunk_bytes(self):
        self.assertTrue(hasattr(self.read, 'chunk_bytes'))

    def test_chunk_bytes(self):
        self.assertIsInstance(self.read.chunk_bytes, int)
        self.assertEqual(32 * 1024 * 1024, self.read.chunk_bytes)

    def test_has_fs(self):
        self.assertTrue(hasattr(self.read, 'fs'))

    def test_fs(self):
        self.assertIsInstance(self.read.fs, MemoryFileSystem)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'
        self.storage = str(Storage.MEMORY)

    def test_empty_path(self):
        read = Reader('', self.storage)
        self.assertEqual('/', read.path)

    def test_root_path(self):
        read = Reader('/', self.storage)
        self.assertEqual('/', read.path)

    def test_path(self):
        read = Reader('/path/to/another/file.txt', self.storage)
        self.assertEqual('/path/to/another/file.txt', read.path)

    def test_path_stripped(self):
        read = Reader(' / path/ ', self.storage)
        self.assertEqual('/path', read.path)

    def test_path_prepended(self):
        read = Reader('path / ', self.storage)
        self.assertEqual('/path', read.path)

    def test_path_raises(self):
        with self.assertRaises(TypeError):
            _ = Reader(1, self.storage)

    def test_storage(self):
        read = Reader(self.path, Storage.FILE)
        self.assertEqual('file', read.storage)

    def test_storage_raises(self):
        with self.assertRaises(ValueError):
            _ = Reader(self.path, 4)

    def test_mode(self):
        read = Reader(self.path, self.storage, mode=Mode.RT)
        self.assertEqual('rt', read.mode)

    def test_mode_raises(self):
        with self.assertRaises(ValueError):
            _ = Reader(self.path, self.storage, mode='wrong')

    def test_chunk_size(self):
        read = Reader(self.path, self.storage, chunk_size=16)
        self.assertEqual(16, read.chunk_size)

    def test_chunk_size_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            _ =  Reader(self.path, self.storage, chunk_size='foo')

    def test_chunk_size_wrong_value_raises(self):
        with self.assertRaises(ValueError):
            _ =  Reader(self.path, self.storage, chunk_size=0)

    def test_storage_kws(self):
        write = Reader(self.path, self.storage, storage_kws={'answer': 42})
        self.assertDictEqual({'answer': 42}, write.storage_kws)

    def test_chunk_bytes_round(self):
        read = Reader(self.path, self.storage, chunk_size=16.2)
        self.assertEqual(64 * 256 * 1024, read.chunk_bytes)
        read = Reader(self.path, self.storage, chunk_size=16.3)
        self.assertEqual(65 * 256 * 1024, read.chunk_bytes)
        read = Reader(self.path, self.storage, chunk_size=16.51)
        self.assertEqual(66 * 256 * 1024, read.chunk_bytes)
        read = Reader(self.path, self.storage, chunk_size=16.8)
        self.assertEqual(67 * 256 * 1024, read.chunk_bytes)
        read = Reader(self.path, self.storage, chunk_size=17.0)
        self.assertEqual(68 * 256 * 1024, read.chunk_bytes)

    def test_fs(self):
        write = Reader(self.path, Storage.FILE)
        self.assertIsInstance(write.fs, LocalFileSystem)


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'
        self.storage = str(Storage.MEMORY)

    def test_has_managed(self):
        write = Reader(self.path, self.storage, overwrite=True)
        self.assertTrue(hasattr(write, '_managed'))
        self.assertTrue(callable(write._managed))

    def test_managed_read(self):
        read = Reader(self.path, self.storage, mode=Mode.RT)

        with read.fs.open(self.path, 'wt') as file:
            file.write('Hello World!')
        self.assertTrue(read.fs.exists(self.path))

        with read._managed(self.path) as file:
            text = file.read()

        self.assertEqual('Hello World!', text)

    def test_managed_open_called_default(self):
        read = Reader(
            self.path,
            self.storage,
            overwrite=True,
            mode=Mode.RT,
            chunk_size=17
        )

        mock_fs = Mock()
        mock_file = mock_open()
        mock_fs.open.return_value = mock_file.return_value

        with patch.object(
            read,
            'fs',
            mock_fs
        ), read._managed(
            '/test/file.txt'
        ):
            pass

        mock_fs.open.assert_called_once_with(
            '/test/file.txt',
            'rt',
            read.chunk_bytes,
            compression=None
        )

    def test_managed_open_called_compression(self):
        read = Reader(self.path, self.storage)

        mock_fs = Mock()
        mock_file = mock_open()
        mock_fs.open.return_value = mock_file.return_value

        with patch.object(
            read,
            'fs',
            mock_fs
        ), read._managed(
            '/test/file.txt',
            Compression.BZ2
        ):
            pass

        mock_fs.open.assert_called_once_with(
            '/test/file.txt',
            'rb',
            read.chunk_bytes,
            compression='bz2'
        )

    def test_managed_open_raises_invalid_compression(self):
        read = Reader(self.path, self.storage)
        with self.assertRaises(ValueError), read._managed('/a/b', 'invalid'):
                pass

    def test_has_non_root(self):
        write = Reader(self.path, self.storage)
        self.assertTrue(hasattr(write, '_non_root'))
        self.assertTrue(callable(write._non_root))

    def test_non_root_empty(self):
        read = Reader(self.path, self.storage)
        path = read._non_root()
        self.assertIsInstance(path, str)
        self.assertEqual(self.path, path)

    def test_non_root_appends(self):
        read = Reader('/path/to/', self.storage)
        path = read._non_root('sub/file.txt')
        self.assertEqual('/path/to/sub/file.txt', path)

    def test_non_root_strips(self):
        read = Reader('/path/to/', self.storage)
        path = read._non_root(' sub/file.txt / ')
        self.assertEqual('/path/to/sub/file.txt', path)

    def test_non_root_replaces(self):
        read = Reader('/path/to/file.txt', self.storage)
        path = read._non_root('/another/different.txt')
        self.assertEqual('/another/different.txt', path)

    def test_non_root_raises_on_root(self):
        read = Reader('/', self.storage)
        with self.assertRaises(ValueError):
            _ = read._non_root('file.txt')


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'

    def test_default_repr(self):
        read = Reader(self.path)
        expected = "Reader('/path/to/file.txt', 'file', 32.0, {})"
        self.assertEqual(expected, repr(read))

    def test_custom_repr(self):
        read = Reader(
            self.path,
            'memory',
            Mode.RT,
            16,
            {'answer': 42},
            'foo',
            bar='baz'
        )
        expected = ("Reader('/path/to/file.txt', 'memory',"
                    " 16.0, {'answer': 42}, 'foo', bar='baz')")
        self.assertEqual(expected, repr(read))

    def test_pickle_works(self):
        read = Reader(self.path)
        _ = pickle.loads(pickle.dumps(read))


if __name__ == '__main__':
    unittest.main()
