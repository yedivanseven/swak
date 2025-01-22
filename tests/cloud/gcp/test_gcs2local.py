import unittest
import pickle
from unittest.mock import patch, Mock
from pathlib import Path
from tempfile import TemporaryDirectory
from swak.cloud.gcp import GcsDir2LocalDir


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.download = GcsDir2LocalDir('project', 'bucket')

    def test_project(self):
        self.assertTrue(hasattr(self.download, 'project'))
        self.assertEqual('project', self.download.project)

    def test_bucket(self):
        self.assertTrue(hasattr(self.download, 'bucket'))
        self.assertEqual('bucket', self.download.bucket)

    def test_prefix(self):
        self.assertTrue(hasattr(self.download, 'prefix'))
        self.assertEqual('', self.download.prefix)

    def test_base_dir(self):
        self.assertTrue(hasattr(self.download, 'base_dir'))
        self.assertEqual('/tmp', self.download.base_dir)

    def test_overwrite(self):
        self.assertTrue(hasattr(self.download, 'overwrite'))
        self.assertFalse(self.download.overwrite)

    def test_skip(self):
        self.assertTrue(hasattr(self.download, 'skip'))
        self.assertFalse(self.download.skip)

    def test_n_threads(self):
        self.assertTrue(hasattr(self.download, 'n_threads'))
        self.assertIsInstance(self.download.n_threads, int)
        self.assertEqual(16, self.download.n_threads)

    def test_chunk_size(self):
        self.assertTrue(hasattr(self.download, 'chunk_size'))
        self.assertIsInstance(self.download.chunk_size, int)
        self.assertEqual(10, self.download.chunk_size)

    def test_kwargs(self):
        self.assertTrue(hasattr(self.download, 'kwargs'))
        self.assertDictEqual({}, self.download.kwargs)

    def test_chunk_bytes(self):
        self.assertTrue(hasattr(self.download, 'chunk_bytes'))
        self.assertIsInstance(self.download.chunk_bytes, int)
        self.assertEqual(10 * 1024 * 1024, self.download.chunk_bytes)

    def test_project_strips(self):
        download = GcsDir2LocalDir(' /.project . /', 'bucket')
        self.assertTrue(hasattr(download, 'project'))
        self.assertEqual('project', download.project)

    def test_bucket_strips(self):
        download = GcsDir2LocalDir('project', '. /bucket./ ')
        self.assertTrue(hasattr(download, 'bucket'))
        self.assertEqual('bucket', download.bucket)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.download = GcsDir2LocalDir(
            'project',
            'bucket',
            'prefix',
            '/base_dir',
            True,
            True,
            8,
            5,
            foo='bar'
        )

    def test_prefix(self):
        self.assertEqual('prefix/', self.download.prefix)

    def test_base_dir(self):
        self.assertEqual('/base_dir', self.download.base_dir)

    def test_overwrite(self):
        self.assertTrue(self.download.overwrite)

    def test_skip(self):
        self.assertTrue(self.download.skip)

    def test_n_threads(self):
        self.assertIsInstance(self.download.n_threads, int)
        self.assertEqual(8, self.download.n_threads)

    def test_chunk_size(self):
        self.assertIsInstance(self.download.chunk_size, int)
        self.assertEqual(5, self.download.chunk_size)

    def test_kwargs(self):
        self.assertDictEqual({'foo': 'bar'}, self.download.kwargs)

    def test_chunk_bytes(self):
        self.assertIsInstance(self.download.chunk_bytes, int)
        self.assertEqual(5 * 1024 * 1024, self.download.chunk_bytes)

    def test_chunk_bytes_truncates(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            chunk_size=5.2
        )
        self.assertIsInstance(download.chunk_bytes, int)
        self.assertEqual(5 * 1024 * 1024, download.chunk_bytes)

    def test_prefix_strips(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            prefix=' ./prefix ./'
        )
        self.assertEqual('prefix/', download.prefix)

    def test_base_dir_strips(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            base_dir='  / base_dir/  '
        )
        self.assertEqual('/base_dir', download.base_dir)

    def test_base_dir_adds(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            base_dir='base_dir'
        )
        self.assertEqual('/base_dir', download.base_dir)


class TestPrefix(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.base_dir = self.tmp.name
        self.blob = Mock()
        self.blob.name = 'blob.parquet'
        self.blob.download_to_file = Mock(return_value=b'Hello World')
        self.bucket = Mock()
        self.bucket.get_blob = Mock(return_value=self.blob)
        self.client_instance = Mock()
        self.client_instance.list_blobs = Mock(return_value=[self.blob])
        self.client_instance.get_bucket = Mock(return_value=self.bucket)
        self.client_patch = patch(
            'google.cloud.storage.Client',
            return_value=self.client_instance
        )
        self.client_class = self.client_patch.start()

    def tearDown(self) -> None:
        self.client_patch.stop()
        self.tmp.cleanup()

    def test_no_prefix(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            base_dir=self.base_dir
        )
        _ = download()
        file = Path(self.base_dir) / 'blob.parquet'
        self.assertTrue(file.exists())
        self.assertTrue(file.is_file())
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix=None
        )

    def test_instantiation_prefix(self):
        self.blob.name = 'pre/fix/blob.parquet'
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            'pre/fix',
            base_dir=self.base_dir
        )
        _ = download()
        directory = Path(self.base_dir) / 'pre' / 'fix'
        self.assertTrue(directory.exists())
        self.assertTrue(directory.is_dir())
        file = directory / 'blob.parquet'
        self.assertTrue(file.exists())
        self.assertTrue(file.is_file())
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='pre/fix/'
        )

    def test_call_prefix(self):
        self.blob.name = 'pre/fix/blob.parquet'
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            base_dir=self.base_dir
        )
        _ = download('pre/fix')
        directory = Path(self.base_dir) / 'pre' / 'fix'
        self.assertTrue(directory.exists())
        self.assertTrue(directory.is_dir())
        file = directory / 'blob.parquet'
        self.assertTrue(file.exists())
        self.assertTrue(file.is_file())
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='pre/fix/'
        )

    def test_partial_prefixes(self):
        self.blob.name = 'pre/fix/blob.parquet'
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            'pre',
            base_dir=self.base_dir
        )
        _ = download('fix')
        directory = Path(self.base_dir) / 'pre' / 'fix'
        self.assertTrue(directory.exists())
        self.assertTrue(directory.is_dir())
        file = directory / 'blob.parquet'
        self.assertTrue(file.exists())
        self.assertTrue(file.is_file())
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='pre/fix/'
        )

    def test_subdirectories_filtered_prefixes(self):
        self.blob.name = 'prefix/foo/blob.parquet'
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            'prefix',
            base_dir=self.base_dir
        )
        _ = download()
        directory = Path(self.base_dir) / 'prefix'
        self.assertTrue(directory.exists())
        self.assertTrue(directory.is_dir())
        self.assertFalse(any(directory.iterdir()))
        file = directory / 'blob.parquet'
        self.assertFalse(file.exists())
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='prefix/'
        )

    def test_local_exists_empty(self):
        self.blob.name = 'prefix/blob.parquet'
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            'prefix',
            base_dir=self.base_dir
        )
        directory = Path(self.base_dir) / 'prefix'
        directory.mkdir()
        _ = download()
        file = directory / 'blob.parquet'
        self.assertTrue(file.exists())
        self.assertTrue(file.is_file())
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='prefix/'
        )

    def test_raises_on_local_exists_non_empty(self):
        self.blob.name = 'prefix/blob.parquet'
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            'prefix',
            base_dir=self.base_dir
        )
        directory = Path(self.base_dir) / 'prefix'
        directory.mkdir()
        file = directory / 'existing.txt'
        with file.open('w') as stream:
            stream.write('Hello World')
        with self.assertRaises(FileExistsError):
            _ = download()


class TestSkipOverwrite(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.base_dir = self.tmp.name
        self.blob = Mock()
        self.blob.name = 'prefix/blob.parquet'
        self.blob.download_to_file = Mock(return_value=b'Hello World')
        self.bucket = Mock()
        self.bucket.get_blob = Mock(return_value=self.blob)
        self.client_instance = Mock()
        self.client_instance.list_blobs = Mock(return_value=[self.blob])
        self.client_instance.get_bucket = Mock(return_value=self.bucket)
        self.client_patch = patch(
            'google.cloud.storage.Client',
            return_value=self.client_instance
        )
        self.client_class = self.client_patch.start()

    def tearDown(self) -> None:
        self.client_patch.stop()
        self.tmp.cleanup()

    def test_skip_true_loads_into_non_existing(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            base_dir=self.base_dir,
            skip=True
        )
        _ = download('prefix')
        directory = Path(self.base_dir) / 'prefix'
        self.assertTrue(directory.exists())
        self.assertTrue(directory.is_dir())
        file = directory / 'blob.parquet'
        self.assertTrue(file.exists())
        self.assertTrue(file.is_file())
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='prefix/'
        )

    def test_skip_true_raises_on_existing_file(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            base_dir=self.base_dir,
            skip=True
        )
        directory = Path(self.base_dir)
        file = directory / 'prefix'
        with file.open('w') as stream:
            stream.write('Hello World')
        with self.assertRaises(FileExistsError):
            _ = download('prefix')

    def test_skip_true_loads_into_existing_empty(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            base_dir=self.base_dir,
            skip=True
        )
        directory = Path(self.base_dir) / 'prefix'
        directory.mkdir()
        _ = download('prefix')
        file = directory / 'blob.parquet'
        self.assertTrue(file.exists())
        self.assertTrue(file.is_file())
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='prefix/'
        )

    def test_skip_true_reads_from_existing_non_empty(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            base_dir=self.base_dir,
            skip=True
        )
        directory = Path(self.base_dir) / 'prefix'
        directory.mkdir()
        file = directory / 'existing.txt'
        with file.open('w') as stream:
            stream.write('Hello World')
        actual = download('prefix')
        self.assertIsInstance(actual, list)
        self.assertEqual(1, len(actual))
        self.assertTrue(actual[0].startswith('/tmp/'))
        self.assertTrue(actual[0].endswith('/prefix/existing.txt'))
        self.client_instance.list_blobs.assert_not_called()

    def test_overwrite_file(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            base_dir=self.base_dir,
            overwrite=True
        )
        directory = Path(self.base_dir) / 'prefix'
        with directory.open('w') as stream:
            stream.write('Hello World')
        _ = download('prefix')
        self.assertTrue(directory.exists())
        self.assertTrue(directory.is_dir())
        file = directory / 'blob.parquet'
        self.assertTrue(file.exists())
        self.assertTrue(file.is_file())
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='prefix/'
        )

    def test_overwrite_directory(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            base_dir=self.base_dir,
            overwrite=True
        )
        directory = Path(self.base_dir) / 'prefix'
        directory.mkdir()
        existing = directory / 'foo.parquet'
        with existing.open('w') as stream:
            stream.write('Hello World')
        _ = download('prefix')
        self.assertTrue(directory.exists())
        self.assertTrue(directory.is_dir())
        self.assertFalse(existing.exists())
        file = directory / 'blob.parquet'
        self.assertTrue(file.exists())
        self.assertTrue(file.is_file())
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='prefix/'
        )


class TestCloud(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.base_dir = self.tmp.name
        self.blob = Mock()
        self.blob.name = 'prefix/blob.parquet'
        self.blob.download_to_file = Mock(return_value=b'Hello World')
        self.blob.chunk_size = 'chunk_size'
        self.bucket = Mock()
        self.bucket.get_blob = Mock(return_value=self.blob)
        self.client_instance = Mock()
        self.client_instance.list_blobs = Mock(return_value=[self.blob])
        self.client_instance.get_bucket = Mock(return_value=self.bucket)
        self.client_patch = patch(
            'google.cloud.storage.Client',
            return_value=self.client_instance
        )
        self.client_class = self.client_patch.start()

    def tearDown(self) -> None:
        self.client_patch.stop()
        self.tmp.cleanup()

    def test_client_called(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            'prefix',
            base_dir=self.base_dir,
        )
        _ = download()
        self.assertEqual(2, self.client_class.call_count)

    def test_client_called_default_args(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            'prefix',
            base_dir=self.base_dir,
        )
        _ = download()
        args1, args2 = self.client_class.call_args_list
        self.assertTupleEqual(('project',), args1[0])
        self.assertDictEqual({}, args1[1])
        self.assertTupleEqual(('project',), args2[0])
        self.assertDictEqual({}, args2[1])

    def test_client_called_kwargs(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            'prefix',
            base_dir=self.base_dir,
            foo='bar'
        )
        _ = download()
        args1, args2 = self.client_class.call_args_list
        self.assertTupleEqual(('project',), args1[0])
        self.assertDictEqual({'foo': 'bar'}, args1[1])
        self.assertTupleEqual(('project',), args2[0])
        self.assertDictEqual({'foo': 'bar'}, args2[1])

    def test_get_bucket_called(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            'prefix',
            base_dir=self.base_dir,
        )
        _ = download()
        self.client_instance.get_bucket.assert_called_once_with('bucket')

    def test_get_blob_called(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            'prefix',
            base_dir=self.base_dir,
        )
        _ = download()
        self.bucket.get_blob.assert_called_once_with('prefix/blob.parquet')

    def test_chunk_size_set(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            'prefix',
            base_dir=self.base_dir,
        )
        _ = download()
        self.assertIsInstance(self.blob.chunk_size, int)
        self.assertEqual(download.chunk_bytes, self.blob.chunk_size)

    def test_download_to_file_called(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            'prefix',
            base_dir=self.base_dir,
        )
        _ = download()
        self.blob.download_to_file.assert_called_once()
        args = self.blob.download_to_file.call_args
        self.assertEqual(2, len(args))
        self.assertEqual(1, len(args[0]))
        self.assertTrue(hasattr(args[0][0], 'name'))
        self.assertTrue(args[0][0].name.startswith('/tmp/'))
        self.assertTrue(args[0][0].name.endswith('/prefix/blob.parquet'))
        self.assertDictEqual({'raw_download': True}, args[1])


class TestMisc(unittest.TestCase):

    def test_repr_default(self):
        download = GcsDir2LocalDir('project', 'bucket')
        expected = ("GcsDir2LocalDir('project', 'bucket', '', "
                    "'/tmp', False, False, 16, 10)")
        self.assertEqual(expected, repr(download))

    def test_repr(self):
        download = GcsDir2LocalDir(
            'project',
            'bucket',
            'prefix',
            '/base_dir',
            True,
            True,
            8,
            5,
            foo='bar'
        )
        expected = ("GcsDir2LocalDir('project', 'bucket', 'prefix/', "
                    "'/base_dir', True, True, 8, 5, foo='bar')")
        self.assertEqual(expected, repr(download))

    def test_pickle_works(self):
        download = GcsDir2LocalDir('project', 'bucket')
        _ = pickle.loads(pickle.dumps(download))


if __name__ == '__main__':
    unittest.main()
