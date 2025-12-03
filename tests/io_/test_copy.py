# import pickle
import unittest
from unittest.mock import patch, Mock
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.implementations.local import LocalFileSystem
from swak.io import Copy, Storage


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.copy = Copy()

    def test_has_src_base(self):
        self.assertTrue(hasattr(self.copy, 'src_base'))

    def test_src_base(self):
        self.assertEqual('/', self.copy.src_base)

    def test_has_tgt_base(self):
        self.assertTrue(hasattr(self.copy, 'tgt_base'))

    def test_tgt_base(self):
        self.assertEqual('/', self.copy.tgt_base)

    def test_has_src_storage(self):
        self.assertTrue(hasattr(self.copy, 'src_storage'))

    def test_src_storage(self):
        self.assertEqual('file', self.copy.src_storage)

    def test_has_tgt_storage(self):
        self.assertTrue(hasattr(self.copy, 'tgt_storage'))

    def test_tgt_storage(self):
        self.assertEqual('file', self.copy.tgt_storage)

    def test_has_overwrite(self):
        self.assertTrue(hasattr(self.copy, 'overwrite'))

    def test_overwrite(self):
        self.assertIsInstance(self.copy.overwrite, bool)
        self.assertFalse(self.copy.overwrite)

    def test_has_skip(self):
        self.assertTrue(hasattr(self.copy, 'skip'))

    def test_skip(self):
        self.assertIsInstance(self.copy.skip, bool)
        self.assertFalse(self.copy.skip)

    def test_has_chunk_size(self):
        self.assertTrue(hasattr(self.copy, 'chunk_size'))

    def test_chunk_size(self):
        self.assertIsInstance(self.copy.chunk_size, float)
        self.assertEqual(32, self.copy.chunk_size)

    def test_has_src_kws(self):
        self.assertTrue(hasattr(self.copy, 'src_kws'))

    def test_src_kws(self):
        self.assertDictEqual({}, self.copy.src_kws)

    def test_has_tgt_kws(self):
        self.assertTrue(hasattr(self.copy, 'tgt_kws'))

    def test_tgt_kws(self):
        self.assertDictEqual({}, self.copy.tgt_kws)

    def test_has_chunk_bytes(self):
        self.assertTrue(hasattr(self.copy, 'chunk_bytes'))

    def test_chunk_bytes(self):
        self.assertIsInstance(self.copy.chunk_bytes, int)
        self.assertEqual(32 * 1024 * 1024, self.copy.chunk_bytes)

    def test_has_src_fs(self):
        self.assertTrue(hasattr(self.copy, 'src_fs'))

    def test_src_fs(self):
        self.assertIsInstance(self.copy.src_fs, LocalFileSystem)

    @patch('fsspec.filesystem')
    def test_src_fs_calls_fsspec_filesystem_with_defaults(self, mock):
        _ = self.copy.src_fs
        mock.assert_called_once_with(self.copy.src_storage)

    def test_has_tgt_fs(self):
        self.assertTrue(hasattr(self.copy, 'tgt_fs'))

    def test_tgt_fs(self):
        self.assertIsInstance(self.copy.tgt_fs, LocalFileSystem)

    @patch('fsspec.filesystem')
    def test_tgt_fs_calls_fsspec_filesystem_with_defaults(self, mock):
        _ = self.copy.tgt_fs
        mock.assert_called_once_with(self.copy.tgt_storage)


class TestAttributes(unittest.TestCase):

    def test_src_base_stripped(self):
        copy = Copy('  / path/to/my   / ')
        self.assertEqual('/path/to/my', copy.src_base)

    def test_src_base_prepended(self):
        copy = Copy('  path/to/my/ ')
        self.assertEqual('/path/to/my', copy.src_base)

    def test_src_base_not_string_raises(self):
        with self.assertRaises(TypeError):
            _ = Copy(123)

    def test_tgt_base_copied(self):
        copy = Copy('  / path/to/my   / ')
        self.assertEqual('/path/to/my', copy.tgt_base)

    def test_tgt_base_explicit_none_copied(self):
        copy = Copy('  / path/to/my   / ', None)
        self.assertEqual('/path/to/my', copy.tgt_base)

    def test_tgt_base_stripped(self):
        copy = Copy('', '  / path/to/my   / ')
        self.assertEqual('/path/to/my', copy.tgt_base)

    def test_tgt_base_prepended(self):
        copy = Copy('', '  path/to/my/ ')
        self.assertEqual('/path/to/my', copy.tgt_base)

    def test_tgt_base_not_string_raises(self):
        with self.assertRaises(TypeError):
            _ = Copy('', 123)

    def test_src_storage(self):
        copy = Copy(src_storage='memory')
        self.assertEqual('memory', copy.src_storage)

    def test_wrong_src_storage_raises(self):
        with self.assertRaises(ValueError):
            _ = Copy(src_storage='wrong')

    def test_tgt_storage_copied(self):
        copy = Copy(src_storage='memory')
        self.assertEqual('memory', copy.tgt_storage)

    def test_tgt_storage_explicit_non_copied(self):
        copy = Copy(src_storage='memory', tgt_storage=None)
        self.assertEqual('memory', copy.tgt_storage)

    def test_tgt_storage(self):
        copy = Copy(tgt_storage='memory')
        self.assertEqual('memory', copy.tgt_storage)

    def test_wrong_tgt_storage_raises(self):
        with self.assertRaises(ValueError):
            _ = Copy(tgt_storage=123)

    def test_overwrite(self):
        copy = Copy(overwrite=True)
        self.assertIsInstance(copy.overwrite, bool)
        self.assertTrue(copy.overwrite)

    def test_skip(self):
        copy = Copy(skip=True)
        self.assertIsInstance(copy.skip, bool)
        self.assertTrue(copy.skip)

    def test_chunk_size(self):
        copy = Copy(chunk_size=16)
        self.assertEqual(16, copy.chunk_size)

    def test_chunk_size_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            _ =  Copy(chunk_size='foo')

    def test_chunk_size_wrong_value_raises(self):
        with self.assertRaises(ValueError):
            _ =  Copy(chunk_size=0)

    def test_src_kws(self):
        copy = Copy(src_kws={'answer': 42})
        self.assertDictEqual({'answer': 42}, copy.src_kws)

    def test_tgt_kws(self):
        copy = Copy(tgt_kws={'answer': 42})
        self.assertDictEqual({'answer': 42}, copy.tgt_kws)

    def test_chunk_bytes_round(self):
        copy = Copy(chunk_size=16.2)
        self.assertEqual(64 * 256 * 1024, copy.chunk_bytes)
        copy = Copy(chunk_size=16.3)
        self.assertEqual(65 * 256 * 1024, copy.chunk_bytes)
        copy = Copy(chunk_size=16.51)
        self.assertEqual(66 * 256 * 1024, copy.chunk_bytes)
        copy = Copy(chunk_size=16.8)
        self.assertEqual(67 * 256 * 1024, copy.chunk_bytes)
        copy = Copy(chunk_size=17.0)
        self.assertEqual(68 * 256 * 1024, copy.chunk_bytes)

    def test_src_fs(self):
        copy = Copy(src_storage=Storage.MEMORY)
        self.assertIsInstance(copy.src_fs, MemoryFileSystem)

    @patch('fsspec.filesystem')
    def test_src_fs_calls_fsspec_filesystem_with_kwargs(self, mock):
        copy = Copy(src_storage=Storage.MEMORY, src_kws={'answer': 42})
        _ = copy.src_fs
        mock.assert_called_once_with(copy.src_storage, answer=42)

    def test_tgt_fs_copied(self):
        copy = Copy(src_storage=Storage.MEMORY)
        self.assertIsInstance(copy.tgt_fs, MemoryFileSystem)

    def test_tgt_fs(self):
        copy = Copy(tgt_storage=Storage.MEMORY)
        self.assertIsInstance(copy.src_fs, LocalFileSystem)
        self.assertIsInstance(copy.tgt_fs, MemoryFileSystem)

    @patch('fsspec.filesystem')
    def test_tgt_fs_calls_fsspec_filesystem_with_kwargs(self, mock):
        copy = Copy(tgt_storage=Storage.MEMORY, tgt_kws={'answer': 42})
        _ = copy.tgt_fs
        mock.assert_called_once_with(copy.tgt_storage, answer=42)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/my/file.txt'
        self.src_base = '/path/to/my'
        self.tgt_base = '/target/path'

    def test_non_root_works(self):
        copy = Copy()
        actual = copy._non_root(self.path)
        self.assertEqual(self.path, actual)

    def test_non_root_raises_on_root(self):
        copy = Copy()
        with self.assertRaises(ValueError):
            _ = copy._non_root('/file.txt')

    def test_non_root_raises_on_dot_dot(self):
        copy = Copy()
        with self.assertRaises(ValueError):
            _ = copy._non_root('/path/to/../my/file.txt')

    def test_src_uri_from_default(self):
        copy = Copy()
        actual = copy._src_uri_from('/path/to/my/file.txt')
        expected = self.src_base + '/file.txt'
        self.assertEqual(expected, actual)

    def test_src_uri_from(self):
        copy = Copy(self.src_base)
        actual = copy._src_uri_from('/path/to/my/file.txt')
        expected = self.src_base + '/file.txt'
        self.assertEqual(expected, actual)

    def test_src_uri_from_normalizes(self):
        copy = Copy(self.src_base)
        actual = copy._src_uri_from('/path/to/./my/file.txt')
        expected = self.src_base + '/file.txt'
        self.assertEqual(expected, actual)

    def test_src_uri_from_appends_root(self):
        copy = Copy(self.src_base)
        actual = copy._src_uri_from('/some/other/file.txt')
        expected = self.src_base + '/some/other/file.txt'
        self.assertEqual(expected, actual)

    def test_src_uri_from_appends_non_root(self):
        copy = Copy(self.src_base)
        actual = copy._src_uri_from('some/other/file.txt')
        expected = self.src_base + '/some/other/file.txt'
        self.assertEqual(expected, actual)

    def test_src_uri_from_calls_non_root(self):
        copy = Copy(self.src_base)
        with patch.object(copy, '_non_root') as mock:
            mock.return_value = 'moch return value'
            _ = copy._src_uri_from('/path/to/./my/file.txt')
            mock.assert_called_once_with('/path/to/my/file.txt')

    def test_tgt_uri_from_default(self):
        copy = Copy(self.src_base)
        actual = copy._tgt_uri_from('/path/to/my/file.txt')
        expected = self.src_base + '/file.txt'
        self.assertEqual(expected, actual)

    def test_tgt_uri_from_replaces(self):
        copy = Copy(self.src_base, self.tgt_base)
        actual = copy._tgt_uri_from('/path/to/my/file.txt')
        expected = self.tgt_base + '/file.txt'
        self.assertEqual(expected, actual)

    def test_tgt_uri_from_calls_non_root(self):
        copy = Copy(self.src_base, self.tgt_base)
        with patch.object(copy, '_non_root') as mock:
            mock.return_value = 'moch return value'
            _ = copy._tgt_uri_from('/path/to/my/file.txt')
            mock.assert_called_once_with('/target/path/file.txt')

    @patch('swak.io.writer.uuid.uuid4')
    def test_tmp(self, uuid):
        uuid.return_value = Mock(hex='hex')
        copy = Copy()
        actual = copy._tmp(self.path)
        expected = self.path + '.tmp.hex'
        self.assertEqual(expected, actual)

    # ToDo: Continue here
    # Test exists/skip(overwrite logic
    # Test parents are created on target
    # Test tmp called
    # Test temporary file created
    # Test temporary file moved
    # Test temporary file removed
    # Test return value

    def test_callable(self):
        copy = Copy()
        self.assertTrue(callable(copy))


if __name__ == '__main__':
    unittest.main()
