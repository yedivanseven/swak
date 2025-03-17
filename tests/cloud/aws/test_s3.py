import unittest
import pickle
from unittest.mock import patch
from swak.cloud.aws import S3


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.s3 = S3()

    def test_has_region_name(self):
        self.assertTrue(hasattr(self.s3, 'region_name'))

    def test_region_name(self):
        self.assertIsNone(self.s3.region_name)

    def test_has_api_version(self):
        self.assertTrue(hasattr(self.s3, 'api_version'))

    def test_api_version(self):
        self.assertIsNone(self.s3.api_version)

    def test_has_use_ssl(self):
        self.assertTrue(hasattr(self.s3, 'use_ssl'))

    def test_use_ssl(self):
        self.assertIsInstance(self.s3.use_ssl, bool)
        self.assertTrue(self.s3.use_ssl)

    def test_has_verify(self):
        self.assertTrue(hasattr(self.s3, 'verify'))

    def test_verify(self):
        self.assertIsInstance(self.s3.verify, bool)
        self.assertTrue(self.s3.verify)

    def test_has_endpoint_url(self):
        self.assertTrue(hasattr(self.s3, 'endpoint_url'))

    def test_endpoint_url(self):
        self.assertIsNone(self.s3.endpoint_url)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.s3, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.s3.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.region_name = ' region  '
        self.api_version = ' api_version  '
        self.use_ssl = False
        self.verify = False
        self.endpoint_url = ' endpoint_url  '
        self.aws_account_id = ' aws_account_id  '
        self.aws_access_key_id = '  aws_access_key_id '
        self.aws_secret_access_key = ' aws_secret_access_key  '
        self.aws_session_token = '  aws_secret_access_key '
        self.kwargs = {'three': 3}
        self.s3 = S3(
            self.region_name,
            self.api_version,
            self.use_ssl,
            self.verify,
            self.endpoint_url,
            self.aws_account_id,
            self.aws_access_key_id,
            self.aws_secret_access_key,
            self.aws_session_token,
            **self.kwargs
        )

    def test_region_name_stripped(self):
        self.assertEqual(self.region_name.strip(), self.s3.region_name)

    def test_api_version_stripped(self):
        self.assertEqual(self.api_version.strip(), self.s3.api_version)

    def test_use_ssl(self):
        self.assertIsInstance(self.s3.use_ssl, bool)
        self.assertFalse(self.s3.use_ssl)

    def test_verify(self):
        self.assertIsInstance(self.s3.verify, bool)
        self.assertFalse(self.s3.verify)

    def test_endpoint_url_stripped(self):
        self.assertEqual(self.endpoint_url.strip(), self.s3.endpoint_url)

    def test_kwargs(self):
        self.assertEqual(self.kwargs, self.s3.kwargs)

    def test_repr(self):
        expected = ("S3('region', 'api_version', False, False, 'endpoint_url',"
                    " aws_account_id='****', aws_access_key_id='****',"
                    " aws_secret_access_key='****', aws_session_token='****',"
                    " three=3)")
        self.assertEqual(expected, repr(self.s3))


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.region_name = 'region'
        self.api_version = 'api_version'
        self.use_ssl = False
        self.verify = False
        self.endpoint_url = 'endpoint_url'
        self.aws_account_id = 'aws_account_id'
        self.aws_access_key_id = 'aws_access_key_id'
        self.aws_secret_access_key = 'aws_secret_access_key'
        self.aws_session_token = 'aws_secret_access_key'
        self.kwargs = {'three': 3}
        self.s3 = S3(
            self.region_name,
            self.api_version,
            self.use_ssl,
            self.verify,
            self.endpoint_url,
            self.aws_account_id,
            self.aws_access_key_id,
            self.aws_secret_access_key,
            self.aws_session_token,
            **self.kwargs
        )

    def test_callable(self):
        self.assertTrue(callable(self.s3))

    @patch('boto3.client')
    @patch('swak.cloud.aws.s3.Config')
    def test_has_client(self, config, client):
        self.assertTrue(hasattr(self.s3, 'client'))

    @patch('boto3.client')
    @patch('swak.cloud.aws.s3.Config')
    def test_client_instantiated_when_requested(self, config, client):
        config.return_value = 'config'
        _ = self.s3.client
        client.assert_called_once_with(
            service_name='s3',
            region_name=self.region_name,
            api_version=self.api_version,
            use_ssl=self.use_ssl,
            verify=self.verify,
            endpoint_url=self.endpoint_url,
            aws_account_id=self.aws_account_id,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            config='config'
        )
        config.assert_called_once_with(**self.kwargs)

    @patch('boto3.client')
    @patch('swak.cloud.aws.s3.Config')
    def test_new_clients_when_requested_again(self, config, client):
        config.return_value = 'config'
        _ = self.s3.client
        _ = self.s3.client
        _ = self.s3.client
        self.assertEqual(3, client.call_count)
        self.assertEqual(3, config.call_count)

    @patch('boto3.client')
    @patch('swak.cloud.aws.s3.Config')
    def test_client_instantiated_when_called(self, config, client):
        config.return_value = 'config'
        _ = self.s3()
        client.assert_called_once_with(
            service_name='s3',
            region_name=self.region_name,
            api_version=self.api_version,
            use_ssl=self.use_ssl,
            verify=self.verify,
            endpoint_url=self.endpoint_url,
            aws_account_id=self.aws_account_id,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            config='config'
        )
        config.assert_called_once_with(**self.kwargs)

    @patch('boto3.client')
    @patch('swak.cloud.aws.s3.Config')
    def test_new_clients_when_called_again(self, config, client):
        config.return_value = 'config'
        _ = self.s3()
        _ = self.s3()
        _ = self.s3()
        self.assertEqual(3, client.call_count)
        self.assertEqual(3, config.call_count)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.s3 = S3()

    def test_default_repr(self):
        expected = ("S3(None, None, True, True, None, aws_account_id=None, "
                    "aws_access_key_id=None, aws_secret_access_key=None, "
                    "aws_session_token=None)")
        self.assertEqual(expected, repr(self.s3))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.s3))


if __name__ == '__main__':
    unittest.main()
