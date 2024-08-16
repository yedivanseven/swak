import unittest
import pickle
from unittest.mock import patch, Mock
from pathlib import Path
import pandas as pd
from swak.cloud.gcp import GcsParquet2DataFrame


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.download = GcsParquet2DataFrame('project', 'bucket')

    def test_project(self):
        self.assertTrue(hasattr(self.download, 'project'))
        self.assertEqual('project', self.download.project)

    def test_bucket(self):
        self.assertTrue(hasattr(self.download, 'bucket'))
        self.assertEqual('bucket', self.download.bucket)

    def test_prefix(self):
        self.assertTrue(hasattr(self.download, 'prefix'))
        self.assertEqual('', self.download.prefix)

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
        download = GcsParquet2DataFrame(' /.project . /', 'bucket')
        self.assertTrue(hasattr(download, 'project'))
        self.assertEqual('project', download.project)

    def test_bucket_strips(self):
        download = GcsParquet2DataFrame('project', '. /bucket./ ')
        self.assertTrue(hasattr(download, 'bucket'))
        self.assertEqual('bucket', download.bucket)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.download = GcsParquet2DataFrame(
            'project',
            'bucket',
            'prefix',
            8,
            5,
            foo='bar'
        )

    def test_prefix(self):
        self.assertEqual('prefix/', self.download.prefix)

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
        download = GcsParquet2DataFrame(
            'project',
            'bucket',
            chunk_size=5.2
        )
        self.assertIsInstance(download.chunk_bytes, int)
        self.assertEqual(5 * 1024 * 1024, download.chunk_bytes)

    def test_prefix_strips(self):
        download = GcsParquet2DataFrame(
            'project',
            'bucket',
            prefix=' ./prefix ./'
        )
        self.assertEqual('prefix/', download.prefix)


class TestPrefix(unittest.TestCase):

    def setUp(self) -> None:
        self.stream = open(Path(__file__).parent / 'file.pqt', 'rb')
        self.blob = Mock()
        self.blob.name = 'blob.pqt'
        self.blob.open = Mock(return_value=self.stream)
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
        self.stream.close()

    def test_no_prefix(self):
        download = GcsParquet2DataFrame(
            'project',
            'bucket'
        )
        _ = download()
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix=None
        )

    def test_instantiation_prefix(self):
        self.blob.name = 'pre/fix/blob.pqt'
        download = GcsParquet2DataFrame(
            'project',
            'bucket',
            'pre/fix'
        )
        _ = download()
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='pre/fix/'
        )

    def test_call_prefix(self):
        self.blob.name = 'pre/fix/blob.pqt'
        download = GcsParquet2DataFrame(
            'project',
            'bucket'
        )
        _ = download('pre/fix')
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='pre/fix/'
        )

    def test_partial_prefixes(self):
        self.blob.name = 'pre/fix/blob.pqt'
        download = GcsParquet2DataFrame(
            'project',
            'bucket',
            'pre'
        )
        _ = download('fix')
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='pre/fix/'
        )

    def test_subdirectories_filtered_prefixes(self):
        self.blob.name = 'prefix/foo/blob.pqt'
        download = GcsParquet2DataFrame(
            'project',
            'bucket',
            'prefix'
        )
        actual = download()
        self.assertIsInstance(actual, pd.DataFrame)
        self.assertTrue(actual.empty)

    def test_instantiation_prefix_interpolated(self):
        self.blob.name = 'pre/foo/fix/blob.pqt'
        download = GcsParquet2DataFrame(
            'project',
            'bucket',
            'pre/{}/fix',
        )
        _ = download('', 'foo')
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='pre/foo/fix/'
        )

    def test_call_prefix_interpolated(self):
        self.blob.name = 'pre/foo/fix/blob.pqt'
        download = GcsParquet2DataFrame(
            'project',
            'bucket',
        )
        _ = download('pre/{}/fix', 'foo')
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='pre/foo/fix/'
        )

    def test_partial_prefixes_interpolated(self):
        self.blob.name = 'pre/foo/bar/fix/blob.pqt'
        download = GcsParquet2DataFrame(
            'project',
            'bucket',
            'pre/{}'
        )
        _ = download('/{}/fix', 'foo', 'bar')
        self.client_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='pre/foo/bar/fix/'
        )


class TestCloud(unittest.TestCase):

    def setUp(self) -> None:
        self.stream = open(Path(__file__).parent / 'file.pqt', 'rb')
        self.blob = Mock()
        self.blob.name = 'prefix/blob.pqt'
        self.blob.open = Mock(return_value=self.stream)
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
        self.stream.close()

    def test_client_called(self):
        download = GcsParquet2DataFrame(
            'project',
            'bucket',
            'prefix'
        )
        _ = download()
        self.assertEqual(2, self.client_class.call_count)

    def test_client_called_default_args(self):
        download = GcsParquet2DataFrame(
            'project',
            'bucket',
            'prefix'
        )
        _ = download()
        args1, args2 = self.client_class.call_args_list
        self.assertTupleEqual(('project',), args1[0])
        self.assertDictEqual({}, args1[1])
        self.assertTupleEqual(('project',), args2[0])
        self.assertDictEqual({}, args2[1])

    def test_client_called_kwargs(self):
        download = GcsParquet2DataFrame(
            'project',
            'bucket',
            'prefix',
            foo='bar'
        )
        _ = download()
        args1, args2 = self.client_class.call_args_list
        self.assertTupleEqual(('project',), args1[0])
        self.assertDictEqual({'foo': 'bar'}, args1[1])
        self.assertTupleEqual(('project',), args2[0])
        self.assertDictEqual({'foo': 'bar'}, args2[1])

    def test_get_bucket_called(self):
        download = GcsParquet2DataFrame(
            'project',
            'bucket',
            'prefix'
        )
        _ = download()
        self.client_instance.get_bucket.assert_called_once_with('bucket')

    def test_get_blob_called(self):
        download = GcsParquet2DataFrame(
            'project',
            'bucket',
            'prefix'
        )
        _ = download()
        self.bucket.get_blob.assert_called_once_with('prefix/blob.pqt')

    def test_blob_open_called(self):
        download = GcsParquet2DataFrame(
            'project',
            'bucket',
            'prefix',
            chunk_size=3
        )
        _ = download()
        self.blob.open.assert_called_once_with(
            'rb',
            chunk_size=download.chunk_bytes,
            raw_download=True
        )

    def test_return_value(self):
        download = GcsParquet2DataFrame('project', 'bucket')
        df = download('prefix')
        expected = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=['A', 'B', 'C']
        )
        self.assertIsInstance(df, pd.DataFrame)
        pd.testing.assert_frame_equal(expected, df)


class TestMisc(unittest.TestCase):

    def test_repr_default(self):
        download = GcsParquet2DataFrame('project', 'bucket')
        expected = "GcsParquet2DataFrame('project', 'bucket', '', 16, 10)"
        self.assertEqual(expected, repr(download))

    def test_repr(self):
        download = GcsParquet2DataFrame(
            'project',
            'bucket',
            'prefix',
            8,
            5,
            foo='bar'
        )
        expected = ("GcsParquet2DataFrame('project', 'bucket', "
                    "'prefix/', 8, 5, foo='bar')")
        self.assertEqual(expected, repr(download))

    def test_pickle_works(self):
        download = GcsParquet2DataFrame('project', 'bucket')
        _ = pickle.dumps(download)


if __name__ == '__main__':
    unittest.main()
