import unittest
import pickle
from unittest.mock import Mock, patch
from swak.cloud.gcp import GcsBucket


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.create = GcsBucket(
            'project',
            'bucket',
            'location'
        )

    def test_project(self):
        self.assertTrue(hasattr(self.create, 'project'))
        self.assertEqual('project', self.create.project)

    def test_bucket(self):
        self.assertTrue(hasattr(self.create, 'bucket'))
        self.assertEqual('bucket', self.create.bucket)

    def test_location(self):
        self.assertTrue(hasattr(self.create, 'location'))
        self.assertEqual('LOCATION', self.create.location)

    def test_blob_expire_days(self):
        self.assertTrue(hasattr(self.create, 'blob_expire_days'))
        self.assertIsNone(self.create.blob_expire_days)

    def test_labels(self):
        self.assertTrue(hasattr(self.create, 'labels'))
        self.assertDictEqual({}, self.create.labels)

    def test_user_project(self):
        self.assertTrue(hasattr(self.create, 'user_project'))
        self.assertEqual('project', self.create.user_project)

    def test_storage_class(self):
        self.assertTrue(hasattr(self.create, 'storage_class'))
        self.assertIsNone(self.create.storage_class)

    def test_requester_pays(self):
        self.assertTrue(hasattr(self.create, 'requester_pays'))
        self.assertFalse(self.create.requester_pays)

    def test_kwargs(self):
        self.assertTrue(hasattr(self.create, 'kwargs'))
        self.assertDictEqual({}, self.create.kwargs)

    def test_project_stripped(self):
        create = GcsBucket(
            ' /.project ./',
            ' /.bucket ./',
            ' location  '
        )
        self.assertEqual('project', create.project)

    def test_bucket_stripped(self):
        create = GcsBucket(
            ' /.project ./',
            ' /.bucket ./',
            ' location  '
        )
        self.assertEqual('bucket', create.bucket)

    def test_location_stripped(self):
        create = GcsBucket(
            ' /.project ./',
            ' /.bucket ./',
            ' location  '
        )
        self.assertEqual('LOCATION', create.location)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.create = GcsBucket(
            'project',
            'bucket',
            'location',
            4,
            {'foo': 'bar'},
            'user',
            'storage',
            True,
            hello='world'
        )

    def test_blob_expire_days(self):
        self.assertIsInstance(self.create.blob_expire_days, int)
        self.assertEqual(4, self.create.blob_expire_days)

    def test_labels(self):
        self.assertDictEqual({'foo': 'bar'}, self.create.labels)

    def test_user_project(self):
        self.assertEqual('user', self.create.user_project)

    def test_storage_class(self):
        self.assertEqual('storage', self.create.storage_class)

    def test_requester_pays(self):
        self.assertTrue(self.create.requester_pays)

    def test_kwargs(self):
        self.assertDictEqual({'hello': 'world'}, self.create.kwargs)

    def test_user_project_stripped(self):
        create = GcsBucket(
            'project',
            'bucket',
            'location',
            user_project=' /. user /.'
        )
        self.assertEqual('user', create.user_project)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.create = GcsBucket(
            'project',
            'bucket',
            'location',
            4,
            {'foo': 'bar'},
            'user',
            'storage',
            True,
            hello='world'
        )

    def test_callable(self):
        self.assertTrue(callable(self.create))

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_client_called_once(self, mock_client, _):
        _ = self.create()
        mock_client.assert_called_once()

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_client_called_once_with_kwargs(self, mock_client, _):
        _ = self.create()
        mock_client.assert_called_once_with('project', hello='world')

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_bucket_called_once(self, _, mock_bucket):
        _ = self.create()
        mock_bucket.assert_called_once()

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_bucket_called_once_with_args(self, mock_client, mock_bucket):
        client = Mock()
        client.get_bucket = Mock()
        mock_client.return_value = client
        _ = self.create()
        mock_bucket.assert_called_once_with(
            client,
            self.create.bucket,
            self.create.user_project
        )

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_bucket_default_attributes_set(self, mock_client, mock_bucket):
        create = GcsBucket(
            'project',
            'bucket',
            'location'
        )
        client = Mock()
        client.get_bucket = Mock()
        mock_client.return_value = client
        bucket = Mock()
        bucket.add_lifecycle_delete_rule = Mock()
        bucket.exists = Mock()
        bucket.create = Mock()
        mock_bucket.return_value = bucket
        _ = create()
        self.assertTrue(hasattr(bucket, 'requester_pays'))
        self.assertFalse(bucket.requester_pays)
        self.assertTrue(hasattr(bucket, 'storage_class'))
        self.assertIsNone(bucket.storage_class)
        self.assertTrue(hasattr(bucket, 'labels'))
        self.assertDictEqual({}, bucket.labels)
        bucket.add_lifecycle_delete_rule.assert_not_called()

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_bucket_attributes_set(self, mock_client, mock_bucket):
        client = Mock()
        client.get_bucket = Mock()
        mock_client.return_value = client
        bucket = Mock()
        bucket.add_lifecycle_delete_rule = Mock()
        bucket.exists = Mock()
        bucket.create = Mock()
        mock_bucket.return_value = bucket
        _ = self.create()
        self.assertTrue(hasattr(bucket, 'requester_pays'))
        self.assertTrue(bucket.requester_pays)
        self.assertTrue(hasattr(bucket, 'storage_class'))
        self.assertEqual('storage', bucket.storage_class)
        self.assertTrue(hasattr(bucket, 'labels'))
        self.assertDictEqual({'foo': 'bar'}, bucket.labels)
        bucket.add_lifecycle_delete_rule.assert_called_once_with(age=4)

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_bucket_exists_called(self, mock_client, mock_bucket):
        client = Mock()
        client.get_bucket = Mock()
        mock_client.return_value = client
        bucket = Mock()
        bucket.add_lifecycle_delete_rule = Mock()
        bucket.exists = Mock()
        bucket.create = Mock()
        mock_bucket.return_value = bucket
        _ = self.create()
        bucket.exists.assert_called_once_with()

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_get_bucket_called_with_defaults(self, mock_client, mock_bucket):
        client = Mock()
        client.get_bucket = Mock()
        mock_client.return_value = client
        bucket = Mock()
        bucket.add_lifecycle_delete_rule = Mock()
        bucket.exists = Mock(return_value=True)
        bucket.create = Mock()
        mock_bucket.return_value = bucket
        _ = self.create()
        client.get_bucket.assert_called_once_with(
            bucket,
            retry=None,
            timeout=None
        )
        bucket.create.assert_not_called()

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_get_bucket_called_with_kwargs(self, mock_client, mock_bucket):
        client = Mock()
        client.get_bucket = Mock()
        mock_client.return_value = client
        bucket = Mock()
        bucket.add_lifecycle_delete_rule = Mock()
        bucket.exists = Mock(return_value=True)
        bucket.create = Mock()
        mock_bucket.return_value = bucket
        _ = self.create(retry='retry', timeout='timeout')
        client.get_bucket.assert_called_once_with(
            bucket,
            retry='retry',
            timeout='timeout'
        )
        bucket.create.assert_not_called()

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_get_bucket_return_value(self, mock_client, mock_bucket):
        client = Mock()
        client.get_bucket = Mock(return_value='old')
        mock_client.return_value = client
        bucket = Mock()
        bucket.add_lifecycle_delete_rule = Mock()
        bucket.exists = Mock(return_value=True)
        bucket.create = Mock()
        mock_bucket.return_value = bucket
        existing, created = self.create()
        self.assertTupleEqual(('old', False), (existing, created))

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_create_called_true_false(self, mock_client, mock_bucket):
        client = Mock()
        mock_client.return_value = client
        bucket = Mock()
        bucket.add_lifecycle_delete_rule = Mock()
        bucket.exists = Mock(return_value=True)
        bucket.create = Mock()
        mock_bucket.return_value = bucket
        _ = self.create(False)
        client.get_bucket.assert_not_called()
        bucket.create.assert_called_once()

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_create_called_false_true(self, mock_client, mock_bucket):
        client = Mock()
        mock_client.return_value = client
        bucket = Mock()
        bucket.add_lifecycle_delete_rule = Mock()
        bucket.exists = Mock(return_value=False)
        bucket.create = Mock()
        mock_bucket.return_value = bucket
        _ = self.create(True)
        client.get_bucket.assert_not_called()
        bucket.create.assert_called_once()

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_create_called_false_false(self, mock_client, mock_bucket):
        client = Mock()
        mock_client.return_value = client
        bucket = Mock()
        bucket.add_lifecycle_delete_rule = Mock()
        bucket.exists = Mock(return_value=False)
        bucket.create = Mock()
        mock_bucket.return_value = bucket
        _ = self.create(False)
        client.get_bucket.assert_not_called()
        bucket.create.assert_called_once()

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_create_called_with_defaults(self, mock_client, mock_bucket):
        client = Mock()
        mock_client.return_value = client
        bucket = Mock()
        bucket.add_lifecycle_delete_rule = Mock()
        bucket.exists = Mock(return_value=False)
        bucket.create = Mock()
        mock_bucket.return_value = bucket
        _ = self.create()
        bucket.create.assert_called_once_with(
            client,
            self.create.project,
            self.create.location,
            retry=None,
            timeout=None
        )

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_create_called_with_kwargs(self, mock_client, mock_bucket):
        client = Mock()
        mock_client.return_value = client
        bucket = Mock()
        bucket.add_lifecycle_delete_rule = Mock()
        bucket.exists = Mock(return_value=False)
        bucket.create = Mock()
        mock_bucket.return_value = bucket
        _ = self.create(retry='retry', timeout='timeout')
        bucket.create.assert_called_once_with(
            client,
            self.create.project,
            self.create.location,
            retry='retry',
            timeout='timeout'
        )

    @patch('swak.cloud.gcp.bucket.Bucket')
    @patch('swak.cloud.gcp.bucket.Client')
    def test_create_return_value(self, mock_client, mock_bucket):
        client = Mock()
        mock_client.return_value = client
        bucket = Mock()
        bucket.add_lifecycle_delete_rule = Mock()
        bucket.exists = Mock(return_value=False)
        bucket.create = Mock()
        mock_bucket.return_value = bucket
        new, created = self.create()
        self.assertIs(bucket, new)
        self.assertTrue(created)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.create = GcsBucket(
            'project',
            'bucket',
            'location',
            4,
            {'foo': 'bar'},
            'user',
            'storage',
            True
        )

    def test_repr(self):
        expected = ("GcsBucket('project', 'bucket', 'LOCATION', "
                    "blob_expire_days=4, labels={'foo': 'bar'}, "
                    "user_project='user', storage_class='storage', "
                    "requester_pays=True)")
        self.assertEqual(expected, repr(self.create))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.create))


if __name__ == '__main__':
    unittest.main()
