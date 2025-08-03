import pickle
import unittest
from unittest.mock import Mock, patch, mock_open
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.implementations.local import LocalFileSystem
from swak.io import Writer, Storage, Mode, Compression


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'
        self.storage = str(Storage.MEMORY)
        self.write = Writer(self.path, Storage.MEMORY)

    def test_has_path(self):
        self.assertTrue(hasattr(self.write, 'path'))

    def test_path(self):
        self.assertEqual(self.path, self.write.path)

    def test_has_storage(self):
        self.assertTrue(hasattr(self.write, 'storage'))

    def test_storage(self):
        self.assertEqual(self.storage, self.write.storage)

    def test_has_overwrite(self):
        self.assertTrue(hasattr(self.write, 'overwrite'))

    def test_overwrite(self):
        self.assertIsInstance(self.write.overwrite, bool)
        self.assertFalse(self.write.overwrite)

    def test_has_skip(self):
        self.assertTrue(hasattr(self.write, 'skip'))

    def test_skip(self):
        self.assertIsInstance(self.write.skip, bool)
        self.assertFalse(self.write.skip)

    def test_has_mode(self):
        self.assertTrue(hasattr(self.write, 'mode'))

    def test_mode(self):
        self.assertEqual('wb', self.write.mode)

    def test_has_chunk_size(self):
        self.assertTrue(hasattr(self.write, 'chunk_size'))

    def test_chunk_size(self):
        self.assertIsInstance(self.write.chunk_size, float)
        self.assertEqual(32.0, self.write.chunk_size)

    def test_has_storage_kws(self):
        self.assertTrue(hasattr(self.write, 'storage_kws'))

    def test_storage_kws(self):
        self.assertDictEqual({}, self.write.storage_kws)

    def test_has_chunk_bytes(self):
        self.assertTrue(hasattr(self.write, 'chunk_bytes'))

    def test_chunk_bytes(self):
        self.assertIsInstance(self.write.chunk_bytes, int)
        self.assertEqual(32 * 1024 * 1024, self.write.chunk_bytes)

    def test_has_fs(self):
        self.assertTrue(hasattr(self.write, 'fs'))

    def test_fs(self):
        self.assertIsInstance(self.write.fs, MemoryFileSystem)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'
        self.storage = str(Storage.MEMORY)

    def test_empty_path(self):
        write = Writer('', self.storage)
        self.assertEqual('/', write.path)

    def test_root_path(self):
        write = Writer('/', self.storage)
        self.assertEqual('/', write.path)

    def test_path(self):
        write = Writer('/path/to/another/file.txt', self.storage)
        self.assertEqual('/path/to/another/file.txt', write.path)

    def test_path_stripped(self):
        write = Writer(' / path/ ', self.storage)
        self.assertEqual('/path', write.path)

    def test_path_prepended(self):
        write = Writer('path / ', self.storage)
        self.assertEqual('/path', write.path)

    def test_path_raises(self):
        with self.assertRaises(TypeError):
            _ = Writer(1, self.storage)

    def test_storage(self):
        write = Writer(self.path, Storage.FILE)
        self.assertEqual('file', write.storage)

    def test_storage_raises(self):
        with self.assertRaises(ValueError):
            _ = Writer(self.path, 4)

    def test_overwrite(self):
        write = Writer(self.path, self.storage, overwrite=True)
        self.assertTrue(write.overwrite)

    def test_skip(self):
        write = Writer(self.path, self.storage, skip=True)
        self.assertTrue(write.skip)

    def test_mode(self):
        write = Writer(self.path, self.storage, mode=Mode.WT)
        self.assertEqual('wt', write.mode)

    def test_mode_raises(self):
        with self.assertRaises(ValueError):
            _ = Writer(self.path, self.storage, mode='wrong')

    def test_chunk_size(self):
        write = Writer(self.path, self.storage, chunk_size=16)
        self.assertEqual(16, write.chunk_size)

    def test_chunk_size_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            _ =  Writer(self.path, self.storage, chunk_size='foo')

    def test_chunk_size_wrong_value_raises(self):
        with self.assertRaises(ValueError):
            _ =  Writer(self.path, self.storage, chunk_size=0)

    def test_storage_kws(self):
        write = Writer(self.path, self.storage, storage_kws={'answer': 42})
        self.assertDictEqual({'answer': 42}, write.storage_kws)

    def test_chunk_bytes_round(self):
        write = Writer(self.path, self.storage, chunk_size=16.2)
        self.assertEqual(64 * 256 * 1024, write.chunk_bytes)
        write = Writer(self.path, self.storage, chunk_size=16.3)
        self.assertEqual(65 * 256 * 1024, write.chunk_bytes)
        write = Writer(self.path, self.storage, chunk_size=16.51)
        self.assertEqual(66 * 256 * 1024, write.chunk_bytes)
        write = Writer(self.path, self.storage, chunk_size=16.8)
        self.assertEqual(67 * 256 * 1024, write.chunk_bytes)
        write = Writer(self.path, self.storage, chunk_size=17.0)
        self.assertEqual(68 * 256 * 1024, write.chunk_bytes)

    def test_fs(self):
        write = Writer(self.path, Storage.FILE)
        self.assertIsInstance(write.fs, LocalFileSystem)


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'
        self.storage = str(Storage.MEMORY)

    def test_has_tmp(self):
        write = Writer(self.path, self.storage)
        self.assertTrue(hasattr(write, '_tmp'))
        self.assertTrue(callable(write._tmp))

    @patch('swak.io.writer.uuid.uuid4')
    def test_tmp(self, uuid):
        uuid.return_value = Mock(hex='hex')
        write = Writer(self.path, self.storage)
        self.assertEqual(self.path + '.tmp.hex', write._tmp(self.path))

    def test_has_uri_from(self):
        write = Writer(self.path, self.storage)
        self.assertTrue(hasattr(write, '_uri_from'))
        self.assertTrue(callable(write._uri_from))

    def test_uri_from_empty(self):
        write = Writer(self.path, self.storage, overwrite=True)
        path = write._uri_from()
        self.assertIsInstance(path, str)
        self.assertEqual(self.path, path)

    def test_uri_from_interpolates(self):
        write = Writer('/path/to/{}.txt', self.storage, overwrite=True)
        path = write._uri_from('file')
        self.assertEqual(self.path, path)

    def test_uri_from_strips(self):
        write = Writer('/path/to/{}.txt {}', self.storage, overwrite=True)
        path = write._uri_from('file', '/')
        self.assertEqual(self.path, path)

    def test_uri_raises_on_root(self):
        write = Writer('/{}.txt', self.storage, overwrite=True)
        with self.assertRaises(ValueError):
            _ = write._uri_from('file')

    def test_uri_from_returns_empty_if_exists_and_skip(self):
        write = Writer('/path/to/{}.txt', self.storage, skip=True)
        write.fs.touch(self.path)
        path = write._uri_from('file')
        self.assertEqual('', path)

    def test_uri_from_raises_if_exists_and_not_overwrite(self):
        write = Writer('/path/to/{}.txt', self.storage)
        write.fs.touch(self.path)
        with self.assertRaises(FileExistsError):
             _ = write._uri_from('file')

    def test_uri_exists_and_overwrite(self):
        write = Writer('/path/to/{}.txt', self.storage, overwrite=True)
        write.fs.touch(self.path)
        path = write._uri_from('file')
        self.assertEqual(self.path, path)

    def test_uri_from_creates_parents(self):
        write = Writer(self.path, self.storage, overwrite=True)
        _ = write._uri_from()
        self.assertTrue(write.fs.exists('/path/to'))
        self.assertTrue(write.fs.isdir('/path/to'))

    def test_has_managed(self):
        write = Writer(self.path, self.storage, overwrite=True)
        self.assertTrue(hasattr(write, '_managed'))
        self.assertTrue(callable(write._managed))

    @patch('swak.io.writer.uuid.uuid4')
    def test_managed_write(self, uuid):
        uuid.return_value = Mock(hex='hex')
        write = Writer(self.path, self.storage, overwrite=True, mode=Mode.WT)
        with write._managed(write.path) as file:
            file.write('Hello World')
        self.assertTrue(write.fs.exists(self.path))
        self.assertFalse(write.fs.exists(self.path + '.tmp.hex'))
        with write.fs.open(self.path, 'rt') as file:
            text = file.read()
        self.assertEqual('Hello World', text)

    @patch('swak.io.writer.uuid.uuid4')
    def test_managed_open_called_default(self, uuid):
        uuid.return_value = Mock(hex='hex')
        write = Writer(
            self.path,
            self.storage,
            overwrite=True,
            mode=Mode.WT,
            chunk_size=17
        )

        mock_fs = Mock()
        mock_file = mock_open()
        mock_fs.open.return_value = mock_file.return_value

        with patch.object(
            write,
            'fs',
            mock_fs
        ), write._managed(
            '/test/file.txt'
        ):
            pass

        mock_fs.open.assert_called_once_with(
            '/test/file.txt.tmp.hex',
            'wt',
            write.chunk_bytes,
            compression=None
        )
        mock_fs.move.assert_called_once_with(
            '/test/file.txt.tmp.hex',
            '/test/file.txt'
        )
        mock_fs.rm.assert_not_called()

    @patch('swak.io.writer.uuid.uuid4')
    def test_managed_open_called_compression(self, uuid):
        uuid.return_value = Mock(hex='hex')
        write = Writer(self.path, self.storage, overwrite=True)

        mock_fs = Mock()
        mock_file = mock_open()
        mock_fs.open.return_value = mock_file.return_value

        with patch.object(
            write,
            'fs',
            mock_fs
        ), write._managed(
            '/test/file.txt',
            Compression.BZ2
        ):
            pass

        mock_fs.open.assert_called_once_with(
            '/test/file.txt.tmp.hex',
            'wb',
            write.chunk_bytes,
            compression='bz2'
        )

    def test_managed_open_raises_invalid_compression(self):
        write = Writer(self.path, self.storage, overwrite=True)

        with self.assertRaises(ValueError), write._managed('/a/b', 'invalid'):
                pass

    @patch('swak.io.writer.uuid.uuid4')
    def test_managed_cleans_up_on_yield_error(self, uuid):
        uuid.return_value = Mock(hex='hex')
        write = Writer(
            self.path,
            self.storage,
            overwrite=True,
            mode=Mode.WT,
            chunk_size=17
        )

        mock_fs = Mock()
        mock_fs.open.side_effect = PermissionError()

        with self.assertRaises(
            PermissionError
        ), patch.object(
            write,
            'fs',
            mock_fs
        ), write._managed(
            '/test/file.txt'
        ):
            pass

        mock_fs.move.assert_not_called()
        mock_fs.rm_assert_not_called()

    @patch('swak.io.writer.uuid.uuid4')
    def test_managed_cleans_up_on_write_error(self, uuid):
        uuid.return_value = Mock(hex='hex')
        write = Writer(
            self.path,
            self.storage,
            overwrite=True,
            mode=Mode.WT,
            chunk_size=17
        )

        mock_fs = Mock()
        mock_file = mock_open()
        mock_fs.open.return_value = mock_file.return_value

        with self.assertRaises(
            ValueError
        ), patch.object(
            write,
            'fs',
            mock_fs
        ), write._managed(
            '/test/file.txt'
        ):
            raise ValueError('Answer is not 48!')

        mock_fs.move.assert_not_called()
        mock_fs.rm.assert_called_once_with('/test/file.txt.tmp.hex')


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.txt'

    def test_default_repr(self):
        write = Writer(self.path)
        expected = ("Writer('/path/to/file.txt', 'file', "
                    "False, False, 32.0, {})")
        self.assertEqual(expected, repr(write))

    def test_custom_repr(self):
        write = Writer(
            self.path,
            'memory',
            True,
            True,
            'wt',
            16.0,
            {'answer': 42},
            'foo',
            bar='baz'
        )
        expected = ("Writer('/path/to/file.txt', 'memory', True, True,"
                    " 16.0, {'answer': 42}, 'foo', bar='baz')")
        self.assertEqual(expected, repr(write))

    def test_pickle_works(self):
        write = Writer(self.path)
        _ = pickle.loads(pickle.dumps(write))


if __name__ == '__main__':
    unittest.main()
