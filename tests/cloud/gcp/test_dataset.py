import unittest
import pickle
import json
from unittest.mock import Mock, patch
from swak.cloud.gcp import DatasetCreator


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.create = DatasetCreator(
            'project',
            'dataset',
            'location'
        )

    def test_has_api_repr(self):
        self.assertTrue(hasattr(self.create, 'api_repr'))

    def test_api_repr(self):
        expected = {
            'datasetReference': {
                'projectId': 'project',
                'datasetId': 'dataset'
            },
            'friendlyName': 'dataset',
            'description': None,
            'defaultTableExpirationMs': None,
            'defaultPartitionExpirationMs': None,
            'labels': {},
            'access': None,
            'location': 'location',
            'isCaseInsensitive': False,
            'defaultCollation': None,
            'defaultRoundingMode': None,
            'maxTimeTravelHours': '168',
            'storageBillingModel': None,
            'resourceTags': {}
        }
        self.assertDictEqual(expected, self.create.api_repr)

    def test_has_to_ms(self):
        self.assertTrue(hasattr(self.create, 'to_ms'))

    def test_callable_to_ms(self):
        self.assertTrue(callable(self.create.to_ms))

    def test_to_ms_none(self):
        actual = self.create.to_ms(None)
        self.assertIsNone(actual)

    def test_to_ms_not_none(self):
        expected = 1000 * 60 * 60 * 24 * 3
        actual = self.create.to_ms(3)
        self.assertEqual(f'{expected}', actual)


class TestAttributes(unittest.TestCase):

    def test_values(self):
        create = DatasetCreator(
            project='project',
            dataset='dataset',
            location='location',
            name='name',
            description='description',
            table_expire_days=14,
            partition_expire_days=7,
            labels={'foo': 'bar'},
            access=[{'user': 'unknown'}],
            case_sensitive=False,
            collation='collation',
            rounding='rounding',
            max_travel_time_hours=48,
            billing='billing',
            tags={'baz': 'pan'}
        )
        expected = {
            "datasetReference": {
                "projectId": "project",
                "datasetId": "dataset"
            },
            "friendlyName": "name",
            "description": "description",
            "defaultTableExpirationMs": "1209600000",
            "defaultPartitionExpirationMs": "604800000",
            "labels": {
                "foo": "bar"
            },
            "access": [
                {
                    "user": "unknown"
                }
            ],
            "location": "location",
            "isCaseInsensitive": True,
            "defaultCollation": "collation",
            "defaultRoundingMode": "rounding",
            "maxTimeTravelHours": "48",
            "storageBillingModel": "billing",
            "resourceTags": {
                "baz": "pan"
            }
        }
        self.assertDictEqual(expected, create.api_repr)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.create = DatasetCreator(
            project='project',
            dataset='dataset',
            location='location',
            name='name',
            description='description',
            table_expire_days=14,
            partition_expire_days=7,
            labels={'foo': 'bar'},
            access=[{'user': 'unknown'}],
            case_sensitive=False,
            collation='collation',
            rounding='rounding',
            max_travel_time_hours=48,
            billing='billing',
            tags={'baz': 'pan'}
        )

    def test_callable(self):
        self.assertTrue(callable(self.create))

    @patch('swak.cloud.gcp.dataset.Client')
    def test_client_called_once(self, mock):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock.return_value = client
        _ = self.create()
        mock.assert_called_once()

    @patch('swak.cloud.gcp.dataset.Client')
    def test_client_called_with_defaults(self, mock):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock.return_value = client
        _ = self.create()
        mock.assert_called_with('project')

    @patch('swak.cloud.gcp.dataset.Client')
    def test_client_called_with_kwargs(self, mock):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock.return_value = client
        _ = self.create(foo='bar')
        mock.assert_called_with('project', foo='bar')

    @patch('swak.cloud.gcp.dataset.Client')
    def test_create_called_once(self, mock):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock.return_value = client
        _ = self.create()
        client.create_dataset.assert_called_once()

    @patch('swak.cloud.gcp.dataset.Dataset')
    @patch('swak.cloud.gcp.dataset.Client')
    def test_dataset_called(self, mock_client, mock_dataset):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock_client.return_value = client
        mock_dataset.from_api_repr = Mock(return_value=self.create.api_repr)
        _ = self.create()
        mock_dataset.from_api_repr.assert_called_once()

    @patch('swak.cloud.gcp.dataset.Dataset')
    @patch('swak.cloud.gcp.dataset.Client')
    def test_dataset_called_with_api_repr(self, mock_client, mock_dataset):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock_client.return_value = client
        mock_dataset.from_api_repr = Mock(return_value=self.create.api_repr)
        _ = self.create()
        mock_dataset.from_api_repr.assert_called_once_with(self.create.api_repr)

    @patch('swak.cloud.gcp.dataset.Dataset')
    @patch('swak.cloud.gcp.dataset.Client')
    def test_create_called_with_defaults(self, mock_client, mock_dataset):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock_client.return_value = client
        mock_dataset.from_api_repr = Mock(return_value=self.create.api_repr)
        _ = self.create()
        client.create_dataset.assert_called_once_with(
            self.create.api_repr,
            True,
            None,
            None
        )

    @patch('swak.cloud.gcp.dataset.Dataset')
    @patch('swak.cloud.gcp.dataset.Client')
    def test_create_called_with_args(self, mock_client, mock_dataset):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock_client.return_value = client
        mock_dataset.from_api_repr = Mock(return_value=self.create.api_repr)
        _ = self.create(False, 'retry', 'timeout')
        client.create_dataset.assert_called_once_with(
            self.create.api_repr,
            False,
            'retry',
            'timeout'
        )

    @patch('swak.cloud.gcp.dataset.Dataset')
    @patch('swak.cloud.gcp.dataset.Client')
    def test_return_value(self, mock_client, mock_dataset):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock_client.return_value = client
        mock_dataset.from_api_repr = Mock(return_value=self.create.api_repr)
        actual = self.create()
        self.assertEqual('success', actual)


class TestMisc(unittest.TestCase):

    def test_repr(self):
        create = DatasetCreator(
            project='project',
            dataset='dataset',
            location='location',
            name='name',
            description='description',
            table_expire_days=14,
            partition_expire_days=7,
            labels={'foo': 'bar'},
            access=[{'user': 'unknown'}],
            case_sensitive=False,
            collation='collation',
            rounding='rounding',
            max_travel_time_hours=48,
            billing='billing',
            tags={'baz': 'pan'}
        )
        self.assertEqual(json.dumps(create.api_repr, indent=4), repr(create))

    def test_pickle_works(self):
        create = DatasetCreator(
            project='project',
            dataset='dataset',
            location='location'
        )
        _ = pickle.dumps(create)


if __name__ == '__main__':
    unittest.main()
