import unittest
import pickle
import json
from unittest.mock import Mock, patch
from google.cloud.exceptions import NotFound
from google.cloud.bigquery import Dataset, DatasetReference
from swak.cloud.gcp import GbqDataset, Gbq, Collation, Billing, Rounding
from swak.cloud.gcp.exceptions import GbqError


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.project = 'project'
        self.dataset = 'dataset'
        self.location = 'europe-north1'
        self.gbq = Mock()
        self.gbq.project = self.project
        self.create = GbqDataset(self.gbq, self.dataset)

    def test_has_gbq(self):
        self.assertTrue(hasattr(self.create, 'gbq'))

    def test_gbq(self):
        self.assertIs(self.create.gbq, self.gbq)

    def test_has_dataset(self):
        self.assertTrue(hasattr(self.create, 'dataset'))

    def test_dataset(self):
        self.assertEqual(self.dataset, self.create.dataset)

    def test_dataset_stripped(self):
        create = GbqDataset(self.gbq, '/ .dataset/ ')
        self.assertEqual('dataset', create.dataset)

    def test_wrong_dataset_raises(self):
        with self.assertRaises(AttributeError):
            _ = GbqDataset(self.gbq, 123)

    def test_has_location(self):
        self.assertTrue(hasattr(self.create, 'location'))

    def test_location(self):
        self.assertEqual(self.location, self.create.location)

    def test_has_exists_ok(self):
        self.assertTrue(hasattr(self.create, 'exists_ok'))

    def test_exists_ok(self):
        self.assertIsInstance(self.create.exists_ok, bool)
        self.assertFalse(self.create.exists_ok)

    def test_has_name(self):
        self.assertTrue(hasattr(self.create, 'name'))

    def test_name(self):
        self.assertEqual(self.dataset, self.create.name)

    def test_name_stripped(self):
        create = GbqDataset(self.gbq, '/ .dataset/ ')
        self.assertEqual('dataset', create.name)

    def test_has_description(self):
        self.assertTrue(hasattr(self.create, 'description'))

    def test_description(self):
        self.assertIsNone(self.create.description)

    def test_has_table_expire_days(self):
        self.assertTrue(hasattr(self.create, 'table_expire_days'))

    def test_table_expire_days(self):
        self.assertIsNone(self.create.table_expire_days)

    def test_has_partition_expire_days(self):
        self.assertTrue(hasattr(self.create, 'partition_expire_days'))

    def test_table_partition_expire_days(self):
        self.assertIsNone(self.create.partition_expire_days)

    def test_has_labels(self):
        self.assertTrue(hasattr(self.create, 'labels'))

    def test_labels(self):
        self.assertDictEqual({}, self.create.labels)

    def test_has_access(self):
        self.assertTrue(hasattr(self.create, 'access'))

    def test_access(self):
        self.assertIsNone(self.create.access)

    def test_has_case_sensitive(self):
        self.assertTrue(hasattr(self.create, 'case_sensitive'))

    def test_case_sensitive(self):
        self.assertIsInstance(self.create.case_sensitive, bool)
        self.assertTrue(self.create.case_sensitive)

    def test_has_collation(self):
        self.assertTrue(hasattr(self.create, 'collation'))

    def test_collation(self):
        self.assertIsNone(self.create.collation)

    def test_has_rounding(self):
        self.assertTrue(hasattr(self.create, 'rounding'))

    def test_rounding(self):
        self.assertIsNone(self.create.rounding)

    def test_has_max_travel_time_hours(self):
        self.assertTrue(hasattr(self.create, 'max_travel_time_hours'))

    def test_max_travel_time_hours(self):
        self.assertIsInstance(self.create.max_travel_time_hours, int)
        self.assertEqual(168, self.create.max_travel_time_hours)

    def test_has_billing(self):
        self.assertTrue(hasattr(self.create, 'billing'))

    def test_billing(self):
        self.assertIsNone(self.create.billing)

    def test_has_tags(self):
        self.assertTrue(hasattr(self.create, 'tags'))

    def test_tags(self):
        self.assertDictEqual({}, self.create.tags)

    def test_has_api_repr(self):
        self.assertTrue(hasattr(self.create, 'api_repr'))

    def test_api_repr(self):
        expected = {
            'datasetReference': {
                'projectId': self.project,
                'datasetId': self.dataset
            },
            'friendlyName': self.dataset,
            'description': None,
            'defaultTableExpirationMs': None,
            'defaultPartitionExpirationMs': None,
            'labels': {},
            'access': None,
            'location': self.location,
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

    def setUp(self):
        self.project = 'project'
        self.gbq = Mock()
        self.gbq.project = self.project
        self.dataset = 'dataset'
        self.location = 'europe-west3'
        self.exists_ok = True
        self.name = 'name'
        self.description = 'description'
        self.table_expire_days = 14
        self.partition_expire_days = 7
        self.labels = {'foo': 'bar'}
        self.access = [{'user': 'unknown'}]
        self.case_sensitive = False
        self.collation = str(Collation.INSENSITIVE)
        self.rounding = str(Rounding.HALF_AWAY)
        self.max_travel_time_hours = 48
        self.billing = str(Billing.LOGICAL)
        self.tags = {'baz': 'pan'}
        self.create = GbqDataset(
            gbq=self.gbq,
            dataset=self.dataset,
            location=self.location,
            exists_ok=self.exists_ok,
            name=self.name,
            description=self.description,
            table_expire_days=self.table_expire_days,
            partition_expire_days=self.partition_expire_days,
            labels=self.labels,
            access=self.access,
            case_sensitive=self.case_sensitive,
            collation=self.collation,
            rounding=self.rounding,
            max_travel_time_hours=self.max_travel_time_hours,
            billing=self.billing,
            tags=self.tags,
        )

    def test_location(self):
        self.assertEqual(self.location, self.create.location)

    def test_location_stripped_lower(self):
        create = GbqDataset(self.gbq, self.dataset, '  LOCATION-01 ')
        self.assertEqual('location-01', create.location)

    def test_exists_ok(self):
        self.assertIsInstance(self.create.exists_ok, bool)
        self.assertTrue(self.create.exists_ok)

    def test_exist_ok_casts(self):
        create = GbqDataset(self.gbq, self.dataset, exists_ok='asd')
        self.assertIsInstance(create.exists_ok, bool)
        self.assertTrue(create.exists_ok)

    def test_name(self):
        self.assertEqual(self.name, self.create.name)

    def test_name_stripped(self):
        create = GbqDataset(self.gbq, self.dataset, name=' name  ')
        self.assertEqual('name', create.name)

    def test_description(self):
        self.assertEqual(self.description, self.create.description)

    def test_description_stripped(self):
        create = GbqDataset(self.gbq, self.dataset, description=' desc  ')
        self.assertEqual('desc', create.description)

    def test_table_expire_days(self):
        self.assertEqual(self.table_expire_days, self.create.table_expire_days)

    def test_table_expire_days_raises_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = GbqDataset(self.gbq, self.dataset, table_expire_days='days')

    def test_table_expire_days_raises_wrong_value(self):
        with self.assertRaises(ValueError):
            _ = GbqDataset(self.gbq, self.dataset, table_expire_days=-3)

    def test_partition_expire_days(self):
        self.assertEqual(
            self.partition_expire_days,
            self.create.partition_expire_days
        )

    def test_partition_expire_days_raises_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = GbqDataset(self.gbq, self.dataset, partition_expire_days='day')

    def test_partition_expire_days_raises_wrong_value(self):
        with self.assertRaises(ValueError):
            _ = GbqDataset(self.gbq, self.dataset, partition_expire_days=0)

    def test_labels(self):
        self.assertDictEqual(self.labels, self.create.labels)

    def test_access(self):
        self.assertListEqual(self.access, self.create.access)

    def test_case_sensitive(self):
        self.assertIsInstance(self.create.case_sensitive, bool)
        self.assertIs(self.case_sensitive, self.create.case_sensitive)

    def test_collation(self):
        self.assertEqual(self.collation, self.create.collation)

    def test_collation_enum(self):
        create = GbqDataset(
            self.gbq,
            self.dataset,
            collation=Collation.SENSITIVE
        )
        self.assertEqual(str(Collation.SENSITIVE), create.collation)

    def test_wrong_collation_raises(self):
        with self.assertRaises(ValueError):
            _ = GbqDataset(self.gbq, self.dataset, collation='collation')

    def test_rounding(self):
        self.assertEqual(self.rounding, self.create.rounding)

    def test_rounding_enum(self):
        create = GbqDataset(
            self.gbq,
            self.dataset,
            rounding=Rounding.HALF_AWAY
        )
        self.assertEqual(str(Rounding.HALF_AWAY), create.rounding)

    def test_wrong_rounding_raises(self):
        with self.assertRaises(ValueError):
            _ = GbqDataset(self.gbq, self.dataset, rounding='rounding')

    def test_max_travel_time_hours(self):
        self.assertEqual(
            self.max_travel_time_hours,
            self.create.max_travel_time_hours
        )

    def test_max_travel_time_hours_raises_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = GbqDataset(
                self.gbq,
                self.dataset,
                max_travel_time_hours='hours'
            )

    def test_max_travel_time_hours_raises_wrong_value(self):
        with self.assertRaises(ValueError):
            _ = GbqDataset(self.gbq, self.dataset, max_travel_time_hours=-5)

    def test_billing(self):
        self.assertEqual(self.billing, self.create.billing)

    def test_billing_enum(self):
        create = GbqDataset(
            self.gbq,
            self.dataset,
            billing=Billing.LOGICAL
        )
        self.assertEqual(str(Billing.LOGICAL), create.billing)

    def test_wrong_billing_raises(self):
        with self.assertRaises(ValueError):
            _ = GbqDataset(self.gbq, self.dataset, billing='billing')

    def test_tags(self):
        self.assertDictEqual(self.tags, self.create.tags)

    def test_api_repr(self):
        expected = {
            "datasetReference": {
                "projectId": self.project,
                "datasetId": self.dataset
            },
            "friendlyName": self.name,
            "description": self.description,
            "defaultTableExpirationMs": self.create.to_ms(
                self.table_expire_days
            ),
            "defaultPartitionExpirationMs": self.create.to_ms(
                self.partition_expire_days
            ),
            "labels": self.labels,
            "access": self.access,
            "location": self.location,
            "isCaseInsensitive": not self.case_sensitive,
            "defaultCollation": self.collation,
            "defaultRoundingMode": self.rounding,
            "maxTimeTravelHours": f"{self.max_travel_time_hours}",
            "storageBillingModel": self.billing,
            "resourceTags": self.tags
        }
        self.assertDictEqual(expected, self.create.api_repr)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.project = 'project'
        self.dataset = 'dataset'
        self.gbq = Mock()
        self.gbq.project = self.project

    def test_callable(self):
        create = GbqDataset(self.gbq, self.dataset)
        self.assertTrue(callable(create))

    @patch('swak.cloud.gcp.dataset.Dataset.from_api_repr')
    def test_dataset_from_api_repr_called_once(self, mock):
        create = GbqDataset(self.gbq, self.dataset, exists_ok=True)
        _ = create()
        mock.assert_called_once_with(create.api_repr)

    def test_client_called_once(self):
        create = GbqDataset(self.gbq, self.dataset, exists_ok=True)
        _ = create()
        self.gbq.assert_called_once()

    def test_get_dataset_called_once(self):
        mock = Mock()
        self.gbq.return_value = mock
        create = GbqDataset(self.gbq, self.dataset, exists_ok=True)
        _ = create()
        mock.get_dataset.assert_called_once()
        arg = mock.get_dataset.call_args[0][0]
        self.assertIsInstance(arg, DatasetReference)
        self.assertEqual(self.project, arg.project)
        self.assertEqual(self.dataset, arg.dataset_id)

    def test_exists_false(self):
        client = Mock()
        client.get_dataset = Mock(side_effect=NotFound('message'))
        self.gbq.return_value = client
        create = GbqDataset(self.gbq, self.dataset, exists_ok=True)
        name, created = create()
        self.assertIsInstance(created, bool)
        self.assertTrue(created)
        self.assertEqual(self.dataset, name)

    def test_exists_true(self):
        client = Mock()
        client.get_dataset = Mock()
        self.gbq.return_value = client
        create = GbqDataset(self.gbq, self.dataset, exists_ok=True)
        name, created = create()
        self.assertIsInstance(created, bool)
        self.assertFalse(created)
        self.assertEqual(self.dataset, name)

    def test_exists_false_does_not_raise(self):
        client = Mock()
        client.get_dataset = Mock(side_effect=NotFound('message'))
        self.gbq.return_value = client
        create = GbqDataset(self.gbq, self.dataset, exists_ok=False)
        _ = create()

    def test_exists_true_raises(self):
        client = Mock()
        client.get_dataset = Mock()
        self.gbq.return_value = client
        create = GbqDataset(self.gbq, self.dataset, exists_ok=False)
        with self.assertRaises(GbqError):
            _ = create()

    def test_create_called(self):
        client = Mock()
        client.create_dataset = Mock()
        client.get_dataset = Mock(side_effect=NotFound('message'))
        self.gbq.return_value = client
        create = GbqDataset(self.gbq, self.dataset, exists_ok=False)
        _ = create()
        client.create_dataset.assert_called_once()
        dataset, exists_ok = client.create_dataset.call_args[0]
        self.assertIs(False, exists_ok)
        self.assertIsInstance(dataset, Dataset)
        self.assertDictEqual(create.api_repr, dataset.to_api_repr())

    def test_create_not_called(self):
        client = Mock()
        client.create_dataset = Mock()
        client.get_dataset = Mock()
        self.gbq.return_value = client
        create = GbqDataset(self.gbq, self.dataset, exists_ok=True)
        _ = create()
        client.create_dataset.assert_not_called()


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.project = 'project'
        self.gbq = Mock()
        self.gbq.project = self.project

    def test_repr(self):
        create = GbqDataset(
            gbq=self.gbq,
            dataset='dataset',
            location='location',
            name='name',
            description='description',
            table_expire_days=14,
            partition_expire_days=7,
            labels={'foo': 'bar'},
            access=[{'user': 'unknown'}],
            case_sensitive=False,
            collation=Collation.SENSITIVE,
            rounding=Rounding.HALF_AWAY,
            max_travel_time_hours=48,
            billing=Billing.LOGICAL,
            tags={'baz': 'pan'}
        )
        self.assertEqual(json.dumps(create.api_repr, indent=4), repr(create))

    def test_pickle_works(self):
        gbq = Gbq('project')
        create = GbqDataset(
            gbq=gbq,
            dataset='dataset',
            location='location'
        )
        _ = pickle.loads(pickle.dumps(create))


if __name__ == '__main__':
    unittest.main()
