import unittest
import pickle
from unittest.mock import patch, Mock
from swak.cloud.aws import S3Parquet2DataFrame


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.s3 = 's3'
        self.bucket = 'bucket'
        self.download = S3Parquet2DataFrame(self.s3, self.bucket)

    def test_has_s3(self):
        self.assertTrue(hasattr(self.download, 's3'))

    def test_s3(self):
        self.assertEqual(self.s3, self.download.s3)

    def test_has_bucket(self):
        self.assertTrue(hasattr(self.download, 'bucket'))

    def test_bucket(self):
        self.assertEqual(self.bucket, self.download.bucket)

    def test_has_prefix(self):
        self.assertTrue(hasattr(self.download, 'prefix'))

    def test_prefix(self):
        self.assertEqual('', self.download.prefix)

    def test_has_bear(self):
        self.assertTrue(hasattr(self.download, 'bear'))

    def test_bear(self):
        self.assertEqual('pandas', self.download.bear)

    def test_has_get_kws(self):
        self.assertTrue(hasattr(self.download, 'get_kws'))

    def test_get_kws(self):
        self.assertDictEqual({}, self.download.get_kws)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.download, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.download.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.s3 = 's3'
        self.bucket = ' / bucket /'
        self.prefix = ' /prefix / '
        self.bear = ' PoLars '
        self.get_kws = {'one': 1}
        self.kwargs = {'two': 2}
        self.download = S3Parquet2DataFrame(
            self.s3,
            self.bucket,
            self.prefix,
            self.bear,
            self.get_kws,
            **self.kwargs
        )

    def test_bucket_stripped(self):
        self.assertEqual(self.bucket.strip(' /'), self.download.bucket)

    def test_prefix_stripped(self):
        self.assertEqual(self.prefix.strip(' /'), self.download.prefix)

    def test_bear_stripped(self):
        self.assertEqual(self.bear.strip().lower(), self.download.bear)

    def test_get_kws(self):
        self.assertDictEqual(self.get_kws, self.download.get_kws)

    def test_kwargs(self):
        self.assertEqual(self.kwargs, self.download.kwargs)

    def test_repr(self):
        expected = ("S3Parquet2DataFrame('s3', 'bucket', 'prefix',"
                    " 'polars', get_kws={'one': 1}, two=2)")
        self.assertEqual(expected, repr(self.download))


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.s3 = Mock()
        self.bucket = 'bucket'
        self.prefix = 'prefix'
        self.bear = 'polars'
        self.get_kws = {'one': 1}
        self.kwargs = {'two': 2}
        self.download = S3Parquet2DataFrame(
            self.s3,
            self.bucket,
            self.prefix,
            self.bear,
            self.get_kws,
            **self.kwargs
        )

    def test_callable(self):
        self.assertTrue(callable(self.download))

    @patch('polars.read_parquet')
    @patch('swak.cloud.aws.s32df.BytesIO')
    def test_get_object_called(self, _, __):
        _ = self.download()
        self.download.s3.client.get_object.assert_called_once_with(
            Key=self.prefix,
            Bucket=self.bucket,
            **self.get_kws
        )

    @patch('polars.read_parquet')
    @patch('swak.cloud.aws.s32df.BytesIO')
    def test_read_parquet_polars_called(self, _, read):
        _ = self.download()
        read.assert_called_once()
        self.assertDictEqual(self.kwargs, read.call_args[1])

    @patch('pandas.read_parquet')
    @patch('swak.cloud.aws.s32df.BytesIO')
    def test_read_parquet_pandas_called(self, _, read):
        download = S3Parquet2DataFrame(
            self.s3,
            self.bucket,
            self.prefix,
            'pandas',
            self.get_kws,
            **self.kwargs
        )
        _ = download()
        read.assert_called_once()
        self.assertDictEqual(self.kwargs, read.call_args[1])

    @patch('pandas.read_parquet')
    @patch('swak.cloud.aws.s32df.BytesIO')
    def test_read_parquet_default_called(self, _, read):
        download = S3Parquet2DataFrame(
            self.s3,
            self.bucket,
            self.prefix,
        )
        _ = download()
        read.assert_called_once()
        self.assertDictEqual({}, read.call_args[1])

    @patch('polars.read_parquet')
    @patch('swak.cloud.aws.s32df.BytesIO')
    def test_return_value(self, _, read):
        read.return_value = 'value'
        actual = self.download()
        self.assertEqual('value', actual)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.s3 = 's3'
        self.bucket = 'bucket'
        self.download = S3Parquet2DataFrame(self.s3, self.bucket)

    def test_default_repr(self):
        expected = ("S3Parquet2DataFrame('s3', 'bucket', '',"
                    " 'pandas', get_kws=None)")
        self.assertEqual(expected, repr(self.download))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.download))


if __name__ == '__main__':
    unittest.main()
