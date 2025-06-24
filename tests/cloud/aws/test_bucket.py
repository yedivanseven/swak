import pickle
import unittest
from unittest.mock import Mock
from botocore.exceptions import ClientError
from swak.cloud.aws import S3, S3Bucket
from swak.cloud.aws.exceptions import S3Error


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.s3 = Mock(spec=S3)
        self.client = Mock()
        self.s3.return_value = self.client
        self.name = 'bucket'
        self.bucket = S3Bucket(self.s3, self.name)

    def test_has_s3(self):
        self.assertTrue(hasattr(self.bucket, 's3'))

    def test_s3(self):
        self.assertIs(self.bucket.s3, self.s3)

    def test_has_bucket(self):
        self.assertTrue(hasattr(self.bucket, 'bucket'))

    def test_bucket(self):
        self.assertEqual(self.name, self.bucket.bucket)

    def test_bucket_stripped(self):
        bucket = S3Bucket(self.s3, ' / bucket/  ')
        self.assertEqual('bucket', bucket.bucket)

    def test_has_location(self):
        self.assertTrue(hasattr(self.bucket, 'location'))

    def test_location(self):
        self.assertEqual('eu-west-1', self.bucket.location)

    def test_has_exists_ok(self):
        self.assertTrue(hasattr(self.bucket, 'exists_ok'))

    def test_exists_ok(self):
        self.assertIsInstance(self.bucket.exists_ok, bool)
        self.assertFalse(self.bucket.exists_ok)

    def test_has_blob_expire_days(self):
        self.assertTrue(hasattr(self.bucket, 'blob_expire_days'))

    def test_blob_expire_days(self):
        self.assertIsNone(self.bucket.blob_expire_days)

    def test_has_bucket_cfg(self):
        self.assertTrue(hasattr(self.bucket, 'bucket_cfg'))

    def test_bucket_cfg(self):
        expected = {
            'CreateBucketConfiguration': {
                'LocationConstraint': 'eu-west-1'
            }
        }
        self.assertDictEqual(expected, self.bucket.bucket_cfg)

    def test_has_lifecycle_cfg(self):
        self.assertTrue(hasattr(self.bucket, 'lifecycle_cfg'))

    def test_lifecycle_cfg(self):
        expected = {
            'Rules': [
                {
                    'ID': 'delete-objects-after-None-days',
                    'Filter': {},
                    'Expiration': {
                        'Days': None,
                    },
                    'Status': 'Enabled'
                }
            ]
        }
        self.assertDictEqual(expected, self.bucket.lifecycle_cfg)


class TesttAttributes(unittest.TestCase):

    def setUp(self):
        self.s3 = Mock(spec=S3)
        self.client = Mock()
        self.s3.return_value = self.client
        self.name = 'bucket'

    def test_location(self):
        bucket = S3Bucket(self.s3, self.name, 'location')
        self.assertEqual('location', bucket.location)

    def test_location_stripped_lowered(self):
        bucket = S3Bucket(self.s3, self.name, ' lOcaTion  ')
        self.assertEqual('location', bucket.location)

    def test_location_in_bucket_cfg(self):
        bucket = S3Bucket(self.s3, self.name, ' lOcaTion  ')
        expected = {
            'CreateBucketConfiguration': {'LocationConstraint': 'location'}
        }
        self.assertDictEqual(expected, bucket.bucket_cfg)

    def test_default_location_bucket_cfg(self):
        bucket = S3Bucket(self.s3, self.name, ' uS-eASt-1  ')
        self.assertDictEqual({}, bucket.bucket_cfg)

    def test_exists_ok(self):
        bucket = S3Bucket(self.s3, self.name, exists_ok=True)
        self.assertIsInstance(bucket.exists_ok, bool)
        self.assertTrue(bucket.exists_ok)

    def test_blob_expire_days(self):
        bucket = S3Bucket(self.s3, self.name, blob_expire_days=3)
        self.assertIsInstance(bucket.blob_expire_days, int)
        self.assertEqual(3, bucket.blob_expire_days)

    def test_raises_on_days_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = S3Bucket(self.s3, self.name, blob_expire_days='three')

    def test_raises_on_days_wrong_value(self):
        with self.assertRaises(ValueError):
            _ = S3Bucket(self.s3, self.name, blob_expire_days=-5)

    def test_lifecycle_cfg(self):
        bucket = S3Bucket(self.s3, self.name, blob_expire_days=3)
        expected = {
            'Rules': [
                {
                    'ID': 'delete-objects-after-3-days',
                    'Filter': {},
                    'Expiration': {
                        'Days': 3,
                    },
                    'Status': 'Enabled'
                }
            ]
        }
        self.assertDictEqual(expected, bucket.lifecycle_cfg)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.s3 = Mock(spec=S3)
        self.client = Mock()
        self.s3.return_value = self.client
        self.name = 'bucket'

    def test_callable(self):
        bucket = S3Bucket(self.s3, self.name)
        self.assertTrue(callable(bucket))

    def test_client_opened(self):
        bucket = S3Bucket(self.s3, self.name, exists_ok=True)
        _ = bucket()
        self.s3.assert_called_once_with()

    def test_head_bucket_called(self):
        bucket = S3Bucket(self.s3, self.name, exists_ok=True)
        _ = bucket()
        self.client.head_bucket.assert_called_once_with(Bucket=bucket.bucket)

    def test_head_bucket_called_with_parts(self):
        bucket = S3Bucket(self.s3, '{}-bucket-{}', exists_ok=True)
        _ = bucket(' ./ head', 'tail. / ')
        self.client.head_bucket.assert_called_once_with(
            Bucket='head-bucket-tail'
        )

    def test_create_bucket_not_called(self):
        bucket = S3Bucket(self.s3, self.name, exists_ok=True)
        _ = bucket()
        self.client.create_bucket.assert_not_called()

    def test_create_bucket_called(self):
        bucket = S3Bucket(self.s3, self.name, exists_ok=True)
        error = ClientError({}, 'head_bucket')
        self.client.head_bucket.side_effect = error
        _ = bucket()
        self.client.create_bucket.assert_called_once_with(
            ACL='private', Bucket=self.name, **bucket.bucket_cfg
        )

    def test_create_bucket_called_with_parts(self):
        bucket = S3Bucket(self.s3, '{}-bucket-{}', exists_ok=True)
        error = ClientError({}, 'head_bucket')
        self.client.head_bucket.side_effect = error
        _ = bucket(' ./ head', 'tail. / ')
        self.client.create_bucket.assert_called_once_with(
            ACL='private', Bucket='head-bucket-tail', **bucket.bucket_cfg
        )

    def test_existing_bucket_raises(self):
        bucket = S3Bucket(self.s3, self.name)
        with self.assertRaises(S3Error):
            _ = bucket()

    def test_non_existing_bucket_does_not_raises(self):
        bucket = S3Bucket(self.s3, self.name)
        error = ClientError({}, 'head_bucket')
        self.client.head_bucket.side_effect = error
        _ = bucket()

    def test_lifecycle_not_called(self):
        bucket = S3Bucket(self.s3, self.name, exists_ok=True)
        _ = bucket()
        self.client.put_bucket_lifecycle_configuration.assert_not_called()

    def test_lifecycle_called(self):
        bucket = S3Bucket(
            self.s3,
            self.name,
            exists_ok=True,
            blob_expire_days=3
        )
        _ = bucket()
        self.client.put_bucket_lifecycle_configuration.assert_called_once_with(
            Bucket=self.name, LifecycleConfiguration=bucket.lifecycle_cfg
        )

    def test_client_closed(self):
        bucket = S3Bucket(self.s3, self.name, exists_ok=True)
        _ = bucket()
        self.client.close.assert_called_once_with()

    def test_return_value_created(self):
        bucket = S3Bucket(self.s3, self.name)
        error = ClientError({}, 'head_bucket')
        self.client.head_bucket.side_effect = error
        actual, created = bucket()
        self.assertEqual(bucket.bucket, actual)
        self.assertIsInstance(created, bool)
        self.assertTrue(created)

    def test_return_value_not_created(self):
        bucket = S3Bucket(self.s3, self.name, exists_ok=True)
        actual, created = bucket()
        self.assertEqual(bucket.bucket, actual)
        self.assertIsInstance(created, bool)
        self.assertFalse(created)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        bucket = S3Bucket('s3', 'bucket')
        expected = "S3Bucket('s3', 'bucket', 'eu-west-1', False, None)"
        self.assertEqual(expected, repr(bucket))

    def test_custom_repr(self):
        bucket = S3Bucket('s3', 'bucket', 'us-east-1', True, 3)
        expected = "S3Bucket('s3', 'bucket', 'us-east-1', True, 3)"
        self.assertEqual(expected, repr(bucket))

    def test_default_pickle_works(self):
        bucket = S3Bucket('s3', 'bucket')
        _ = pickle.loads(pickle.dumps(bucket))

    def test_custom_pickle_works(self):
        bucket = S3Bucket('s3', 'bucket', 'us-east-1', True, 3)
        _ = pickle.loads(pickle.dumps(bucket))


if __name__ == '__main__':
    unittest.main()
