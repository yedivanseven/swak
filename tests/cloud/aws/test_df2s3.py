import io
import unittest
import pickle
from unittest.mock import patch, Mock
from swak.cloud.aws import DataFrame2S3Parquet


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.s3 = 's3'
        self.bucket = 'bucket'
        self.upload = DataFrame2S3Parquet(self.s3, self.bucket)

    def test_has_s3(self):
        self.assertTrue(hasattr(self.upload, 's3'))

    def test_s3(self):
        self.assertEqual(self.s3, self.upload.s3)

    def test_has_bucket(self):
        self.assertTrue(hasattr(self.upload, 'bucket'))

    def test_bucket(self):
        self.assertEqual(self.bucket, self.upload.bucket)

    def test_has_prefix(self):
        self.assertTrue(hasattr(self.upload, 'prefix'))

    def test_prefix(self):
        self.assertEqual('', self.upload.prefix)

    def test_has_extra_kws(self):
        self.assertTrue(hasattr(self.upload, 'extra_kws'))

    def test_extra_kws(self):
        self.assertDictEqual({}, self.upload.extra_kws)

    def test_has_upload_kws(self):
        self.assertTrue(hasattr(self.upload, 'upload_kws'))

    def test_upload_kws(self):
        self.assertDictEqual({}, self.upload.upload_kws)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.upload, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.upload.kwargs)

    def test_has_client(self):
        upload = DataFrame2S3Parquet(Mock(), self.bucket)
        self.assertTrue(hasattr(upload, 'client'))

    def test_client(self):
        s3 = Mock()
        client = Mock()
        s3.client = client
        upload = DataFrame2S3Parquet(s3, self.bucket)
        self.assertIs(upload.client, client)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.s3 = 's3'
        self.bucket = ' / bucket /'
        self.prefix = ' /prefix / '
        self.extra_kws = {'one': 1}
        self.upload_kws = {'two': 2}
        self.kwargs = {'three': 3}
        self.upload = DataFrame2S3Parquet(
            self.s3,
            self.bucket,
            self.prefix,
            self.extra_kws,
            self.upload_kws,
            **self.kwargs
        )

    def test_bucket_stripped(self):
        self.assertEqual(self.bucket.strip(' /'), self.upload.bucket)

    def test_prefix_stripped(self):
        self.assertEqual(self.prefix.strip().lstrip('/'), self.upload.prefix)

    def test_extra_kws(self):
        self.assertDictEqual(self.extra_kws, self.upload.extra_kws)

    def test_upload_kws(self):
        self.assertDictEqual(self.upload_kws, self.upload.upload_kws)

    def test_kwargs(self):
        self.assertEqual(self.kwargs, self.upload.kwargs)

    def test_repr(self):
        expected = ("DataFrame2S3Parquet('s3', 'bucket', 'prefix /', "
                    "extra_kws={'one': 1}, upload_kws={'two': 2}, three=3)")
        self.assertEqual(expected, repr(self.upload))


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.s3 = Mock()
        self.bucket = 'bucket'
        self.prefix = 'prefix'
        self.extra_kws = {'one': 1}
        self.upload_kws = {'two': 2}
        self.kwargs = {'three': 3}
        self.upload = DataFrame2S3Parquet(
            self.s3,
            self.bucket,
            self.prefix,
            self.extra_kws,
            self.upload_kws,
            **self.kwargs
        )

    def test_callable(self):
        self.assertTrue(callable(self.upload))

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_to_parquet_called(self, _):
        df = Mock()
        _ = self.upload(df)
        df.to_parquet.assert_called_once()
        buffer = df.to_parquet.call_args[0][0]
        kwargs = df.to_parquet.call_args[1]
        self.assertIsInstance(buffer, io.BytesIO)
        self.assertDictEqual(self.kwargs, kwargs)

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_transfer_config_called(self, config):
        df = Mock()
        _ = self.upload(df)
        config.assert_called_once_with(**self.upload_kws)

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_upload_fileobj_called(self, config_cls):
        df = Mock()
        config = Mock()
        config_cls.return_value = config
        _ = self.upload(df)
        self.upload.s3.client.upload_fileobj.assert_called_once()
        args = self.upload.s3.client.upload_fileobj.call_args[0]
        self.assertTupleEqual((), args)
        kwargs = self.upload.s3.client.upload_fileobj.call_args[1]
        file_obj = kwargs.pop('Fileobj')
        self.assertIsInstance(file_obj, io.BytesIO)
        expected = {
            'Bucket': self.bucket,
            'Key': self.prefix,
            'ExtraArgs': self.extra_kws,
            'Config': config
        }
        self.assertDictEqual(expected, kwargs)

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_prefix_interpolated_stripped(self, _):
        df = Mock()
        s3 = Mock()
        upload = DataFrame2S3Parquet(s3, 'bucket', ' This {} is {}!  ')
        _ = upload(df, 'class', 'great')
        key = s3.client.upload_fileobj.call_args[1].pop('Key')
        self.assertTrue('This class is great!', key)

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_return_value(self, _):
        df = Mock()
        actual = self.upload(df)
        self.assertTupleEqual((), actual)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.s3 = 's3'
        self.bucket = 'bucket'
        self.upload = DataFrame2S3Parquet(self.s3, self.bucket)

    def test_default_repr(self):
        expected = ("DataFrame2S3Parquet('s3', 'bucket', '',"
                    " extra_kws=None, upload_kws=None)")
        self.assertEqual(expected, repr(self.upload))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.upload))


if __name__ == '__main__':
    unittest.main()
