import unittest
import pickle
from unittest.mock import patch, Mock
from swak.cloud.gcp import DataFrame2GcsParquet, Gcs
from swak.cloud.gcp.exceptions import GcsError


# ToDo: Continue here!
class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.gcs = 'gcs'
        self.bucket = 'bucket'
        self.upload = DataFrame2GcsParquet(self.gcs, self.bucket)

    def test_has_gcs(self):
        self.assertTrue(hasattr(self.upload, 'gcs'))

    def test_gcs(self):
        self.assertEqual(self.gcs, self.upload.gcs)

    def test_has_bucket(self):
        self.assertTrue(hasattr(self.upload, 'bucket'))

    def test_bucket(self):
        self.assertEqual(self.bucket, self.upload.bucket)

    def test_has_prefix(self):
        self.assertTrue(hasattr(self.upload, 'prefix'))

    def test_prefix(self):
        self.assertEqual('', self.upload.prefix)

    def test_has_overwrite(self):
        self.assertTrue(hasattr(self.upload, 'overwrite'))

    def test_overwrite(self):
        self.assertIsInstance(self.upload.overwrite, bool)
        self.assertFalse(self.upload.overwrite)

    def test_has_skip(self):
        self.assertTrue(hasattr(self.upload, 'skip'))

    def test_skip(self):
        self.assertIsInstance(self.upload.skip, bool)
        self.assertFalse(self.upload.skip)

    def test_has_chunk_size(self):
        self.assertTrue(hasattr(self.upload, 'chunk_size'))

    def test_chunk_size(self):
        self.assertIsInstance(self.upload.chunk_size, int)
        self.assertEqual(40, self.upload.chunk_size)

    def test_has_chunk_bytes(self):
        self.assertTrue(hasattr(self.upload, 'chunk_bytes'))

    def test_chunk_bytes(self):
        self.assertIsInstance(self.upload.chunk_bytes, int)
        self.assertEqual(40 * 1024 * 1024, self.upload.chunk_bytes)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.upload, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.upload.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.s3 = 's3'
        self.bucket = ' / bucket/ '
        self.prefix = ' /prefix '
        self.overwrite = True
        self.skip = True
        self.chunk_size = 12
        self.kwargs = {'three': 3}
        self.upload = DataFrame2GcsParquet(
            self.s3,
            self.bucket,
            self.prefix,
            self.overwrite,
            self.skip,
            self.chunk_size,
            **self.kwargs
        )

    def test_bucket_stripped(self):
        self.assertEqual(self.bucket.strip(' /'), self.upload.bucket)

    def test_prefix_stripped(self):
        self.assertEqual(self.prefix.strip().lstrip('/'), self.upload.prefix)

    def test_overwrite(self):
        self.assertTrue(self.upload.overwrite)

    def test_skip(self):
        self.assertTrue(self.upload.skip)

    def test_chunk_size(self):
        self.assertEqual(self.chunk_size, self.upload.chunk_size)

    def test_chunk_bytes(self):
        self.assertEqual(12 * 1024 * 1024, self.upload.chunk_bytes)

    def test_kwargs(self):
        self.assertEqual(self.kwargs, self.upload.kwargs)

if __name__ == '__main__':
    unittest.main()
