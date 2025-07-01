import io
import unittest
import pickle
from botocore.exceptions import ClientError
from unittest.mock import patch, Mock
from swak.cloud.aws import DataFrame2S3Parquet, S3
from swak.cloud.aws.exceptions import S3Error


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


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.s3 = 's3'
        self.bucket = ' / bucket/ '
        self.prefix = ' /prefix '
        self.overwrite = True
        self.skip = True
        self.extra_kws = {'one': 1}
        self.upload_kws = {'two': 2}
        self.kwargs = {'three': 3}
        self.upload = DataFrame2S3Parquet(
            self.s3,
            self.bucket,
            self.prefix,
            self.overwrite,
            self.skip,
            self.extra_kws,
            self.upload_kws,
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

    def test_extra_kws(self):
        self.assertDictEqual(self.extra_kws, self.upload.extra_kws)

    def test_upload_kws(self):
        self.assertDictEqual(self.upload_kws, self.upload.upload_kws)

    def test_kwargs(self):
        self.assertEqual(self.kwargs, self.upload.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.s3 = Mock(spec=S3)
        self.client = Mock()
        self.s3.return_value = self.client
        self.bucket = 'bucket'
        self.prefix = 'prefix'
        self.skip = False
        self.overwrite = True
        self.extra_kws = {'one': 1}
        self.upload_kws = {'two': 2}
        self.kwargs = {'three': 3}
        self.upload = DataFrame2S3Parquet(
            self.s3,
            self.bucket,
            self.prefix,
            self.overwrite,
            self.skip,
            extra_kws=self.extra_kws,
            upload_kws=self.upload_kws,
            **self.kwargs
        )

    def test_callable(self):
        self.assertTrue(callable(self.upload))

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_client_created(self, _):
        df = Mock()
        _ = self.upload(df)
        self.s3.assert_called_once_with()

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_head_object_called(self, _):
        df = Mock()
        _ = self.upload(df)
        self.client.head_object.assert_called_once_with(
            Bucket=self.bucket, Key='prefix'
        )

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_to_parquet_called_object_not_exists(self, _):
        self.client.head_object.side_effect = ClientError({}, 'head_object')
        df = Mock()
        _ = self.upload(df)
        df.to_parquet.assert_called_once()
        df.write_parquet.assert_not_called()
        buffer = df.to_parquet.call_args[0][0]
        kwargs = df.to_parquet.call_args[1]
        self.assertIsInstance(buffer, io.BytesIO)
        self.assertDictEqual(self.kwargs, kwargs)

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_write_parquet_called_object_not_exists(self, _):
        self.client.head_object.side_effect = ClientError({}, 'head_object')
        df = Mock(spec=['write_parquet'])
        _ = self.upload(df)
        df.write_parquet.assert_called_once()
        buffer = df.write_parquet.call_args[0][0]
        kwargs = df.write_parquet.call_args[1]
        self.assertIsInstance(buffer, io.BytesIO)
        self.assertDictEqual(self.kwargs, kwargs)

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_transfer_config_called_object_not_exists(self, config):
        self.client.head_object.side_effect = ClientError({}, 'head_object')
        df = Mock()
        _ = self.upload(df)
        config.assert_called_once_with(**self.upload_kws)

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_upload_fileobj_called_object_not_exists(self, config_cls):
        self.client.head_object.side_effect = ClientError({}, 'head_object')
        df = Mock()
        config = Mock()
        config_cls.return_value = config
        _ = self.upload(df)
        self.client.upload_fileobj.assert_called_once()
        args = self.client.upload_fileobj.call_args[0]
        self.assertTupleEqual((), args)
        kwargs = self.client.upload_fileobj.call_args[1]
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
    def test_client_closed_object_not_exists(self, _):
        self.client.head_object.side_effect = ClientError({}, 'head_object')
        df = Mock()
        _ = self.upload(df)
        self.client.close.assert_called_once_with()

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_raises_object_exists(self, config):
        df = Mock()
        upload = DataFrame2S3Parquet(
            self.s3,
            'bucket',
            'prefix',
            skip=False,
            overwrite=False,
        )
        with self.assertRaises(S3Error):
            _ = upload(df)
        self.client.close.assert_called_once_with()
        df.to_parquet.assert_not_called()
        df.write_parquet.assert_not_called()
        self.client.upload_fileobj.assert_not_called()
        config.assert_not_called()

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_skip_object_exists_overwrite_false(self, config):
        df = Mock()
        upload = DataFrame2S3Parquet(
            self.s3,
            'bucket',
            'prefix',
            skip=True,
            overwrite=False,
        )
        _ = upload(df)
        self.client.close.assert_called_once_with()
        df.to_parquet.assert_not_called()
        df.write_parquet.assert_not_called()
        self.client.upload_fileobj.assert_not_called()
        config.assert_not_called()

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_skip_object_exists_overwrite_true(self, config):
        df = Mock()
        upload = DataFrame2S3Parquet(
            self.s3,
            'bucket',
            'prefix',
            skip=True,
            overwrite=True,
        )
        _ = upload(df)
        self.client.close.assert_called_once_with()
        df.to_parquet.assert_not_called()
        df.write_parquet.assert_not_called()
        self.client.upload_fileobj.assert_not_called()
        config.assert_not_called()

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_to_parquet_called_object_exists(self, _):
        df = Mock()
        _ = self.upload(df)
        df.to_parquet.assert_called_once()
        df.write_parquet.assert_not_called()
        buffer = df.to_parquet.call_args[0][0]
        kwargs = df.to_parquet.call_args[1]
        self.assertIsInstance(buffer, io.BytesIO)
        self.assertDictEqual(self.kwargs, kwargs)

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_write_parquet_called_object_exists(self, _):
        df = Mock(spec=['write_parquet'])
        _ = self.upload(df)
        df.write_parquet.assert_called_once()
        buffer = df.write_parquet.call_args[0][0]
        kwargs = df.write_parquet.call_args[1]
        self.assertIsInstance(buffer, io.BytesIO)
        self.assertDictEqual(self.kwargs, kwargs)

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_transfer_config_called_object_exists(self, config):
        df = Mock()
        _ = self.upload(df)
        config.assert_called_once_with(**self.upload_kws)

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_upload_fileobj_called_object_exists(self, config_cls):
        df = Mock()
        config = Mock()
        config_cls.return_value = config
        _ = self.upload(df)
        self.client.upload_fileobj.assert_called_once()
        args = self.client.upload_fileobj.call_args[0]
        self.assertTupleEqual((), args)
        kwargs = self.client.upload_fileobj.call_args[1]
        file_obj = kwargs.pop('Fileobj')
        self.assertIsInstance(file_obj, io.BytesIO)
        expected = {
            'Bucket': self.bucket,
            'Key': self.prefix,
            'ExtraArgs': self.extra_kws,
            'Config': config,
        }
        self.assertDictEqual(expected, kwargs)

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_client_closed_object_exists(self, _):
        df = Mock()
        _ = self.upload(df)
        self.client.close.assert_called_once_with()

    @patch('swak.cloud.aws.df2s3.TransferConfig')
    def test_prefix_interpolated_stripped(self, _):
        df = Mock()
        upload = DataFrame2S3Parquet(
            self.s3,
            'bucket',
            ' ./This {} is {}! . ',
            skip=False,
            overwrite=True,
        )
        _ = upload(df, 'class', 'great')
        key = self.client.upload_fileobj.call_args[1].pop('Key')
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
        expected = ("DataFrame2S3Parquet('s3', 'bucket', '', False,"
                    " False, extra_kws=None, upload_kws=None)")
        self.assertEqual(expected, repr(self.upload))

    def test_repr(self):
        upload = DataFrame2S3Parquet(
            self.s3,
            self.bucket,
            'prefix',
            True,
            True,
            extra_kws={'one': 1},
            upload_kws={'two': 2},
            three=3
        )
        expected = ("DataFrame2S3Parquet('s3', 'bucket', 'prefix', True, True,"
                    " extra_kws={'one': 1}, upload_kws={'two': 2}, three=3)")
        self.assertEqual(expected, repr(upload))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.upload))


if __name__ == '__main__':
    unittest.main()
