import unittest
import pickle
import json
from unittest.mock import Mock, patch
from google.cloud.exceptions import NotFound
from swak.cloud.gcp import GbqDataset


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.create = GbqDataset(
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

    def test_kwargs(self):
        self.assertTrue(hasattr(self.create, 'kwargs'))
        self.assertDictEqual({}, self.create.kwargs)

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

    def test_project_stripped(self):
        create = GbqDataset(
            ' / project . /',
            '/ .dataset/ ',
            ' location'
        )
        self.assertEqual('project', create.project)

    def test_dataset_stripped(self):
        create = GbqDataset(
            ' / project . /',
            '/ .dataset/ ',
            ' location'
        )
        self.assertEqual('dataset', create.dataset)

    def test_location_stripped(self):
        create = GbqDataset(
            ' / project . /',
            '/ .dataset/ ',
            ' location'
        )
        self.assertEqual('location', create.location)

    def test_name_stripped(self):
        create = GbqDataset(
            ' / project . /',
            '/ .dataset/ ',
            ' location'
        )
        self.assertEqual('dataset', create.name)


class TestAttributes(unittest.TestCase):

    def test_values(self):
        create = GbqDataset(
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
            tags={'baz': 'pan'},
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

    def test_kwargs(self):
        create = GbqDataset(
            'project',
            'dataset',
            'location',
            hello='world'
        )
        self.assertDictEqual({'hello': 'world'}, create.kwargs)

    def test_name_stripped(self):
        create = GbqDataset(
            ' / project . /',
            '/ .dataset/ ',
            ' location',
            name=' name  '
        )
        self.assertEqual('name', create.name)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.create = GbqDataset(
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
            tags={'baz': 'pan'},
            hello='world'
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
    def test_client_called_with_kwargs(self, mock):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock.return_value = client
        _ = self.create()
        mock.assert_called_with('project', location='location', hello='world')

    @patch('swak.cloud.gcp.dataset.Client')
    def test_create_called_once(self, mock):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock.return_value = client
        _ = self.create()
        client.create_dataset.assert_called_once()

    @patch('swak.cloud.gcp.dataset.Dataset')
    @patch('swak.cloud.gcp.dataset.Client')
    def test_dataset_called(self, mock_client, mock_dataset_cls):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock_client.return_value = client
        mock_dataset_obj = Mock()
        mock_dataset_cls.from_api_repr = Mock(return_value=mock_dataset_obj)
        _ = self.create()
        mock_dataset_cls.from_api_repr.assert_called_once()

    @patch('swak.cloud.gcp.dataset.Dataset')
    @patch('swak.cloud.gcp.dataset.Client')
    def test_dataset_called_with_api_repr(self, mock_client, mock_dataset_cls):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock_client.return_value = client
        mock_dataset_obj = Mock()
        mock_dataset_cls.from_api_repr = Mock(return_value=mock_dataset_obj)
        _ = self.create()
        mock_dataset_cls.from_api_repr.assert_called_once_with(
            self.create.api_repr
        )

    @patch('swak.cloud.gcp.dataset.Dataset')
    @patch('swak.cloud.gcp.dataset.Client')
    def test_get_called_with_defaults(self, mock_client, mock_dataset_cls):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock_client.return_value = client
        mock_dataset_obj = Mock()
        mock_dataset_obj.reference = 'reference'
        mock_dataset_cls.from_api_repr = Mock(return_value=mock_dataset_obj)
        _ = self.create()
        client.get_dataset.assert_called_once_with(
            'reference',
            None,
            None
        )

    @patch('swak.cloud.gcp.dataset.Dataset')
    @patch('swak.cloud.gcp.dataset.Client')
    def test_get_called_with_args(self, mock_client, mock_dataset_cls):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock_client.return_value = client
        mock_dataset_obj = Mock()
        mock_dataset_obj.reference = 'reference'
        mock_dataset_cls.from_api_repr = Mock(return_value=mock_dataset_obj)
        _ = self.create(False, 'retry', 'timeout')
        client.get_dataset.assert_called_once_with(
            'reference',
            'retry',
            'timeout'
        )

    @patch('swak.cloud.gcp.dataset.Dataset')
    @patch('swak.cloud.gcp.dataset.Client')
    def test_create_called_with_defaults(self, mock_client, mock_dataset_cls):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock_client.return_value = client
        mock_dataset_obj = Mock()
        mock_dataset_cls.from_api_repr = Mock(return_value=mock_dataset_obj)
        _ = self.create()
        client.create_dataset.assert_called_once_with(
            mock_dataset_obj,
            True,
            None,
            None
        )

    @patch('swak.cloud.gcp.dataset.Dataset')
    @patch('swak.cloud.gcp.dataset.Client')
    def test_create_called_with_args(self, mock_client, mock_dataset_cls):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock_client.return_value = client
        mock_dataset_obj = Mock()
        mock_dataset_cls.from_api_repr = Mock(return_value=mock_dataset_obj)
        _ = self.create(False, 'retry', 'timeout')
        client.create_dataset.assert_called_once_with(
            mock_dataset_obj,
            False,
            'retry',
            'timeout'
        )

    @patch('swak.cloud.gcp.dataset.Dataset')
    @patch('swak.cloud.gcp.dataset.Client')
    def test_return_value_new(self, mock_client, mock_dataset_cls):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        mock_client.return_value = client
        mock_dataset_obj = Mock()
        mock_dataset_cls.from_api_repr = Mock(return_value=mock_dataset_obj)
        existing, created = self.create()
        self.assertTupleEqual(('success', True), (existing, created))

    @patch('swak.cloud.gcp.dataset.Dataset')
    @patch('swak.cloud.gcp.dataset.Client')
    def test_return_value_existing(self, mock_client, mock_dataset_cls):
        client = Mock()
        client.create_dataset = Mock(return_value='success')
        client.get_dataset.side_effect = NotFound('error')
        mock_client.return_value = client
        mock_dataset_obj = Mock()
        mock_dataset_cls.from_api_repr = Mock(return_value=mock_dataset_obj)
        existing, created = self.create()
        self.assertTupleEqual(('success', False), (existing, created))


class TestMisc(unittest.TestCase):

    def test_repr(self):
        create = GbqDataset(
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
        create = GbqDataset(
            project='project',
            dataset='dataset',
            location='location'
        )
        _ = pickle.dumps(create)


if __name__ == '__main__':
    unittest.main()
