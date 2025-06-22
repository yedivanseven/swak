import unittest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError
from swak.cloud.aws import S3
from swak.cloud.aws import S3Bucket
from swak.cloud.aws.exceptions import S3Error


# Todo: Actually write unit tests inspired by these here!
class TestS3Bucket(unittest.TestCase):
    """Comprehensive test suite for S3Bucket class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_s3 = Mock(spec=S3)
        self.mock_client = Mock()
        self.mock_s3.return_value = self.mock_client

        # Default test parameters
        self.bucket_name = 'test-bucket'
        self.location = 'eu-west-1'

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        bucket = S3Bucket(self.mock_s3, self.bucket_name)

        self.assertEqual(bucket.s3, self.mock_s3)
        self.assertEqual(bucket.bucket, self.bucket_name)
        self.assertEqual(bucket.location, 'eu-west-1')  # default
        self.assertFalse(bucket.exists_ok)
        self.assertIsNone(bucket.blob_expire_days)

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        bucket = S3Bucket(
            self.mock_s3,
            '  my-bucket  ',
            location='  US-WEST-2  ',
            exists_ok=True,
            blob_expire_days=30,
        )

        self.assertEqual(bucket.bucket, 'my-bucket')
        self.assertEqual(bucket.location, 'us-west-2')  # should be lowercased
        self.assertTrue(bucket.exists_ok)
        self.assertEqual(bucket.blob_expire_days, 30)

    def test_init_bucket_name_stripping(self):
        """Test that bucket names are properly stripped of spaces and slashes."""
        bucket = S3Bucket(self.mock_s3, '  /test-bucket/  ')
        self.assertEqual(bucket.bucket, 'test-bucket')

    def test_init_invalid_blob_expire_days_type(self):
        """Test that invalid blob_expire_days type raises TypeError."""
        with self.assertRaises(TypeError) as cm:
            S3Bucket(
                self.mock_s3, self.bucket_name, blob_expire_days='invalid'
            )

        self.assertIn('blob_expire_days', str(cm.exception))
        self.assertIn('convertible to integer', str(cm.exception))

    def test_init_invalid_blob_expire_days_value(self):
        """Test that blob_expire_days < 1 raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            S3Bucket(self.mock_s3, self.bucket_name, blob_expire_days=0)

        self.assertIn('greater (or equal) to one', str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            S3Bucket(self.mock_s3, self.bucket_name, blob_expire_days=-5)

    def test_init_blob_expire_days_edge_cases(self):
        """Test blob_expire_days with edge case values."""
        # Test minimum valid value
        bucket = S3Bucket(self.mock_s3, self.bucket_name, blob_expire_days=1)
        self.assertEqual(bucket.blob_expire_days, 1)

        # Test float conversion
        bucket = S3Bucket(
            self.mock_s3, self.bucket_name, blob_expire_days=30.7
        )
        self.assertEqual(bucket.blob_expire_days, 30)

    def test_bucket_cfg_non_us_east_1(self):
        """Test bucket_cfg property for non-us-east-1 regions."""
        bucket = S3Bucket(self.mock_s3, self.bucket_name, location='eu-west-1')
        expected = {
            'CreateBucketConfiguration': {'LocationConstraint': 'eu-west-1'}
        }
        self.assertEqual(bucket.bucket_cfg, expected)

    def test_bucket_cfg_us_east_1(self):
        """Test bucket_cfg property for us-east-1 region."""
        bucket = S3Bucket(self.mock_s3, self.bucket_name, location='us-east-1')
        self.assertEqual(bucket.bucket_cfg, {})

    def test_lifecycle_cfg(self):
        """Test lifecycle_cfg property."""
        bucket = S3Bucket(self.mock_s3, self.bucket_name, blob_expire_days=30)
        expected = {
            'Rules': [
                {
                    'ID': 'delete-objects-after-30-days',
                    'Filter': {},
                    'Expiration': {
                        'Days': 30,
                    },
                    'Status': 'Enabled',
                }
            ]
        }
        self.assertEqual(bucket.lifecycle_cfg, expected)

    def test_call_bucket_already_exists_ok(self):
        """Test calling with existing bucket when exists_ok=True."""
        bucket = S3Bucket(self.mock_s3, self.bucket_name, exists_ok=True)

        # Mock successful head_bucket (bucket exists)
        self.mock_client.head_bucket.return_value = {}

        result_bucket, was_created = bucket()

        self.assertEqual(result_bucket, self.bucket_name)
        self.assertFalse(was_created)  # Bucket existed, wasn't created
        self.mock_client.head_bucket.assert_called_once_with(
            Bucket=self.bucket_name
        )
        self.mock_client.create_bucket.assert_not_called()
        self.mock_client.close.assert_called_once()

    def test_call_bucket_already_exists_not_ok(self):
        """Test calling with existing bucket when exists_ok=False."""
        bucket = S3Bucket(self.mock_s3, self.bucket_name, exists_ok=False)

        # Mock successful head_bucket (bucket exists)
        self.mock_client.head_bucket.return_value = {}

        with self.assertRaises(S3Error) as cm:
            bucket()

        self.assertIn(self.bucket_name, str(cm.exception))
        self.assertIn('already exists', str(cm.exception))
        self.mock_client.create_bucket.assert_not_called()

    def test_call_bucket_does_not_exist(self):
        """Test calling when bucket doesn't exist - should create it."""
        bucket = S3Bucket(self.mock_s3, self.bucket_name, location='eu-west-1')

        # Mock head_bucket raising ClientError (bucket doesn't exist)
        self.mock_client.head_bucket.side_effect = ClientError(
            {'Error': {'Code': '404', 'Message': 'Not Found'}}, 'head_bucket'
        )
        self.mock_client.create_bucket.return_value = {}

        result_bucket, was_created = bucket()

        self.assertEqual(result_bucket, self.bucket_name)
        self.assertTrue(was_created)  # Bucket was created
        self.mock_client.head_bucket.assert_called_once_with(
            Bucket=self.bucket_name
        )
        self.mock_client.create_bucket.assert_called_once_with(
            ACL='private',
            Bucket=self.bucket_name,
            CreateBucketConfiguration={'LocationConstraint': 'eu-west-1'},
        )
        self.mock_client.close.assert_called_once()

    def test_call_bucket_does_not_exist_us_east_1(self):
        """Test bucket creation in us-east-1 region (no CreateBucketConfiguration)."""
        bucket = S3Bucket(self.mock_s3, self.bucket_name, location='us-east-1')

        # Mock head_bucket raising ClientError
        self.mock_client.head_bucket.side_effect = ClientError(
            {'Error': {'Code': '404', 'Message': 'Not Found'}}, 'head_bucket'
        )
        self.mock_client.create_bucket.return_value = {}

        result_bucket, was_created = bucket()

        self.assertTrue(was_created)
        self.mock_client.create_bucket.assert_called_once_with(
            ACL='private', Bucket=self.bucket_name
        )

    def test_call_with_lifecycle_configuration(self):
        """Test calling with lifecycle configuration."""
        bucket = S3Bucket(self.mock_s3, self.bucket_name, blob_expire_days=7)

        # Mock bucket doesn't exist
        self.mock_client.head_bucket.side_effect = ClientError(
            {'Error': {'Code': '404', 'Message': 'Not Found'}}, 'head_bucket'
        )
        self.mock_client.create_bucket.return_value = {}
        self.mock_client.put_bucket_lifecycle_configuration.return_value = {}

        bucket()

        self.mock_client.put_bucket_lifecycle_configuration.assert_called_once_with(
            Bucket=self.bucket_name,
            LifecycleConfiguration={
                'Rules': [
                    {
                        'ID': 'delete-objects-after-7-days',
                        'Filter': {},
                        'Expiration': {'Days': 7},
                        'Status': 'Enabled',
                    }
                ]
            },
        )

    def test_call_without_lifecycle_configuration(self):
        """Test calling without lifecycle configuration."""
        bucket = S3Bucket(
            self.mock_s3, self.bucket_name
        )  # No blob_expire_days

        # Mock bucket doesn't exist
        self.mock_client.head_bucket.side_effect = ClientError(
            {'Error': {'Code': '404', 'Message': 'Not Found'}}, 'head_bucket'
        )
        self.mock_client.create_bucket.return_value = {}

        bucket()

        self.mock_client.put_bucket_lifecycle_configuration.assert_not_called()

    def test_call_with_string_interpolation(self):
        """Test calling with string interpolation in bucket name."""
        bucket = S3Bucket(self.mock_s3, 'test-{}-{}', location='eu-west-1')

        # Mock bucket doesn't exist
        self.mock_client.head_bucket.side_effect = ClientError(
            {'Error': {'Code': '404', 'Message': 'Not Found'}}, 'head_bucket'
        )
        self.mock_client.create_bucket.return_value = {}

        result_bucket, was_created = bucket('env', 'v1')

        self.assertEqual(result_bucket, 'test-env-v1')
        self.assertTrue(was_created)
        self.mock_client.head_bucket.assert_called_once_with(
            Bucket='test-env-v1'
        )
        self.mock_client.create_bucket.assert_called_once_with(
            ACL='private',
            Bucket='test-env-v1',
            CreateBucketConfiguration={'LocationConstraint': 'eu-west-1'},
        )

    def test_call_with_string_interpolation_and_stripping(self):
        """Test string interpolation with spaces/slashes that need stripping."""
        bucket = S3Bucket(self.mock_s3, 'test-{}-{}')

        self.mock_client.head_bucket.side_effect = ClientError(
            {'Error': {'Code': '404', 'Message': 'Not Found'}}, 'head_bucket'
        )

        result_bucket, was_created = bucket('  env  ', '  /v1/  ')

        self.assertEqual(
            result_bucket, 'test-  env  -  /v1/'
        )  # Format first, then strip
        self.mock_client.head_bucket.assert_called_once_with(
            Bucket='test-  env  -  /v1'
        )

    def test_call_client_error_propagation(self):
        """Test that unexpected ClientErrors from create_bucket are propagated."""
        bucket = S3Bucket(self.mock_s3, self.bucket_name)

        # Mock head_bucket indicating bucket doesn't exist
        self.mock_client.head_bucket.side_effect = ClientError(
            {'Error': {'Code': '404', 'Message': 'Not Found'}}, 'head_bucket'
        )

        # Mock create_bucket raising an unexpected error
        create_error = ClientError(
            {
                'Error': {
                    'Code': 'BucketAlreadyOwnedByYou',
                    'Message': 'Your previous request to create the named bucket succeeded',
                }
            },
            'create_bucket',
        )
        self.mock_client.create_bucket.side_effect = create_error

        with self.assertRaises(ClientError):
            bucket()

    def test_s3_client_lifecycle(self):
        """Test that S3 client is properly created and closed."""
        bucket = S3Bucket(self.mock_s3, self.bucket_name, exists_ok=True)

        # Mock bucket exists
        self.mock_client.head_bucket.return_value = {}

        bucket()

        self.mock_s3.assert_called_once()  # Client created
        self.mock_client.close.assert_called_once()  # Client closed

    def test_to_int_static_method_none(self):
        """Test __to_int static method with None input."""
        result = S3Bucket._S3Bucket__to_int(None)
        self.assertIsNone(result)

    def test_to_int_static_method_valid_int(self):
        """Test __to_int static method with valid integers."""
        self.assertEqual(S3Bucket._S3Bucket__to_int(5), 5)
        self.assertEqual(S3Bucket._S3Bucket__to_int(1), 1)
        self.assertEqual(S3Bucket._S3Bucket__to_int(100), 100)

    def test_to_int_static_method_valid_string(self):
        """Test __to_int static method with valid string."""
        self.assertEqual(S3Bucket._S3Bucket__to_int('42'), 42)

    def test_to_int_static_method_valid_float(self):
        """Test __to_int static method with valid float."""
        self.assertEqual(S3Bucket._S3Bucket__to_int(3.14), 3)
        self.assertEqual(S3Bucket._S3Bucket__to_int(5.9), 5)

    def test_integration_full_workflow(self):
        """Integration test for complete workflow."""
        bucket = S3Bucket(
            self.mock_s3,
            'integration-test-{}',
            location='us-west-2',
            exists_ok=False,
            blob_expire_days=30,
        )

        # Mock bucket doesn't exist
        self.mock_client.head_bucket.side_effect = ClientError(
            {'Error': {'Code': '404', 'Message': 'Not Found'}}, 'head_bucket'
        )
        self.mock_client.create_bucket.return_value = {}
        self.mock_client.put_bucket_lifecycle_configuration.return_value = {}

        result_bucket, was_created = bucket('prod')

        # Verify results
        self.assertEqual(result_bucket, 'integration-test-prod')
        self.assertTrue(was_created)

        # Verify call sequence
        self.mock_client.head_bucket.assert_called_once_with(
            Bucket='integration-test-prod'
        )
        self.mock_client.create_bucket.assert_called_once_with(
            ACL='private',
            Bucket='integration-test-prod',
            CreateBucketConfiguration={'LocationConstraint': 'us-west-2'},
        )
        self.mock_client.put_bucket_lifecycle_configuration.assert_called_once()
        self.mock_client.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
