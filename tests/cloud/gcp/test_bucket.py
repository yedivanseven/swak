import unittest
import pickle
from unittest.mock import Mock
from swak.cloud.gcp.exceptions import GcsError
from swak.cloud.gcp import GcsBucket, Gcs


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.project = 'project'
        self.gcs = Mock()
        self.gcs.project = self.project
        self.create = GcsBucket(self.gcs, 'bucket')

    def test_gcs(self):
        self.assertTrue(hasattr(self.create, 'gcs'))
        self.assertIs(self.create.gcs, self.gcs)

    def test_bucket(self):
        self.assertTrue(hasattr(self.create, 'bucket'))
        self.assertEqual('bucket', self.create.bucket)

    def test_bucket_stripped(self):
        create = GcsBucket(self.gcs, ' /.bucket ./')
        self.assertEqual('bucket', create.bucket)

    def test_location(self):
        self.assertTrue(hasattr(self.create, 'location'))
        self.assertEqual('EUROPE-NORTH1', self.create.location)

    def test_exists_ok(self):
        self.assertTrue(hasattr(self.create, 'exists_ok'))
        self.assertIsInstance(self.create.exists_ok, bool)
        self.assertFalse(self.create.exists_ok)

    def test_age(self):
        self.assertTrue(hasattr(self.create, 'age'))
        self.assertIsNone(self.create.age)

    def test_user_project(self):
        self.assertTrue(hasattr(self.create, 'user_project'))
        self.assertEqual('project', self.create.user_project)

    def test_requester_pays(self):
        self.assertTrue(hasattr(self.create, 'requester_pays'))
        self.assertIsInstance(self.create.requester_pays, bool)
        self.assertFalse(self.create.requester_pays)

    def test_kwargs(self):
        self.assertTrue(hasattr(self.create, 'kwargs'))
        self.assertDictEqual({}, self.create.kwargs)

    def test_lifecycle(self):
        self.assertTrue(hasattr(self.create, 'lifecycle'))
        expected = {'action': {'type': 'Delete'}, 'condition': {'age': None}}
        self.assertDictEqual(expected, self.create.lifecycle)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.project = 'project'
        self.gcs = Mock()
        self.gcs.project = self.project
        self.create = GcsBucket(
            self.gcs,
            'bucket',
            ' location ',
            True,
            2,
            ' /.user_project ./',
            True,
            hello='world',
            answer=42
        )

    def test_location(self):
        self.assertEqual('LOCATION', self.create.location)

    def test_exists_ok(self):
        self.assertIsInstance(self.create.exists_ok, bool)
        self.assertTrue(self.create.exists_ok)

    def test_age(self):
        self.assertIsInstance(self.create.age, int)
        self.assertEqual(2, self.create.age)

    def test_raises_one_wrong_age_type(self):
        with self.assertRaises(TypeError):
            GcsBucket(self.gcs, 'bucket', age='foo')

    def test_raises_one_wrong_age_value(self):
        with self.assertRaises(ValueError):
            GcsBucket(self.gcs, 'bucket', age=0)

    def test_user_project(self):
        self.assertEqual('user_project', self.create.user_project)

    def test_requester_pays(self):
        self.assertIsInstance(self.create.requester_pays, bool)
        self.assertTrue(self.create.requester_pays)

    def test_kwargs(self):
        expected = {'hello': 'world', 'answer': 42}
        self.assertDictEqual(expected, self.create.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.lifecycle_rules = [
            {'action': {'type': 'Delete'}, 'condition': {'age': 17}},
            {'action': {'type': 'Forget'}, 'condition': {'feeling': 'good'}},
            {'reaction': {'type': 'Reply'}, 'reason': {'answer': 42}}
        ]
        self.bucket = Mock(spec=[
            'lifecycle_rules',
            'name',
            'location',
            'patch',
            'foo',
            'baz'
        ])
        self.bucket.lifecycle_rules = self.lifecycle_rules
        self.bucket.name = 'bucket'
        self.bucket.location = 'LOCATION'
        self.client = Mock()
        self.client.lookup_bucket = Mock(return_value=self.bucket)
        self.client.get_bucket = Mock(return_value=self.bucket)
        self.gcs = Mock(return_value=self.client)
        self.project = 'project'
        self.gcs.project = self.project

    def test_callable(self):
        create = GcsBucket(self.gcs, 'bucket')
        self.assertTrue(callable(create))

    def test_client_created(self):
        self.client.lookup_bucket.return_value = None
        create = GcsBucket(self.gcs, 'bucket')
        _ = create()
        self.gcs.assert_called_once_with()

    def test_lookup_bucket_called(self):
        self.client.lookup_bucket.return_value = None
        create = GcsBucket(self.gcs, 'bucket')
        _ = create()
        self.client.lookup_bucket.assert_called_once_with('bucket')

    def test_lookup_bucket_interpolated_parts(self):
        self.client.lookup_bucket.return_value = None
        create = GcsBucket(self.gcs, '{}_bucket_{}')
        _ = create(' . / a', 'full / .')
        self.client.lookup_bucket.assert_called_once_with('a_bucket_full')

    def test_raises_on_existing(self):
        create = GcsBucket(self.gcs, 'bucket')
        with self.assertRaises(GcsError):
            _ = create()
        self.client.create_bucket.assert_not_called()

    def test_passes_on_existing_exists_ok(self):
        create = GcsBucket(self.gcs, 'bucket', exists_ok=True)
        _ = create()
        self.client.create_bucket.assert_not_called()

    def test_create_bucket_called_default(self):
        self.client.lookup_bucket.return_value = None
        create = GcsBucket(self.gcs, 'bucket')
        _ = create()
        self.client.create_bucket.assert_called_once_with(
            'bucket',
            create.requester_pays,
            self.gcs.project,
            create.user_project,
            create.location
        )

    def test_create_bucket_interpolated_parts(self):
        self.client.lookup_bucket.return_value = None
        create = GcsBucket(self.gcs, '{}_bucket_{}')
        _ = create(' . / a', 'full / .')
        self.client.create_bucket.assert_called_once_with(
            'a_bucket_full',
            create.requester_pays,
            self.gcs.project,
            create.user_project,
            create.location
        )

    def test_create_bucket_called_custom(self):
        self.client.lookup_bucket.return_value = None
        create = GcsBucket(
            self.gcs,
            'bucket',
            'location',
            user_project='user_project',
            requester_pays=True
        )
        _ = create()
        self.client.create_bucket.assert_called_once_with(
            'bucket',
            True,
            self.gcs.project,
            'user_project',
            'LOCATION'
        )

    def test_get_bucket_called(self):
        self.client.lookup_bucket.return_value = None
        create = GcsBucket(self.gcs, 'bucket')
        _ = create()
        self.client.get_bucket.assert_called_once_with('bucket')

    def test_get_bucket_interpolated_parts(self):
        self.client.lookup_bucket.return_value = None
        create = GcsBucket(self.gcs, '{}_bucket_{}')
        _ = create(' . / a', 'full / .')
        self.client.get_bucket.assert_called_once_with('a_bucket_full')

    def test_lifecycle_rules_unchanged_on_lookup(self):
        create = GcsBucket(self.gcs, 'bucket', exists_ok=True)
        _ = create()
        self.assertListEqual(self.lifecycle_rules, self.bucket.lifecycle_rules)

    def test_lifecycle_rules_none_one_added_on_lookup(self):
        self.bucket.lifecycle_rules = None
        create = GcsBucket(self.gcs, 'bucket', exists_ok=True, age=2)
        _ = create()
        self.assertListEqual([create.lifecycle], self.bucket.lifecycle_rules)

    def test_lifecycle_rules_empty_one_added_on_lookup(self):
        self.bucket.lifecycle_rules = []
        create = GcsBucket(self.gcs, 'bucket', exists_ok=True, age=2)
        _ = create()
        self.assertListEqual([create.lifecycle], self.bucket.lifecycle_rules)

    def test_lifecycle_rules_changed_on_lookup(self):
        create = GcsBucket(self.gcs, 'bucket', exists_ok=True, age=2)
        _ = create()
        expected = [*self.lifecycle_rules[1:], create.lifecycle]
        self.assertListEqual(expected, self.bucket.lifecycle_rules)

    def test_lifecycle_rules_unchanged_on_create(self):
        self.client.lookup_bucket.return_value = None
        create = GcsBucket(self.gcs, 'bucket')
        _ = create()
        self.assertListEqual(self.lifecycle_rules, self.bucket.lifecycle_rules)

    def test_lifecycle_rules_none_one_added_on_create(self):
        self.client.lookup_bucket.return_value = None
        self.bucket.lifecycle_rules = None
        create = GcsBucket(self.gcs, 'bucket', age=2)
        _ = create()
        self.assertListEqual([create.lifecycle], self.bucket.lifecycle_rules)

    def test_lifecycle_rules_empty_one_added_on_create(self):
        self.client.lookup_bucket.return_value = None
        self.bucket.lifecycle_rules = []
        create = GcsBucket(self.gcs, 'bucket', age=2)
        _ = create()
        self.assertListEqual([create.lifecycle], self.bucket.lifecycle_rules)

    def test_lifecycle_rules_changed_on_create(self):
        self.client.lookup_bucket.return_value = None
        create = GcsBucket(self.gcs, 'bucket', age=2)
        _ = create()
        expected = [*self.lifecycle_rules[1:], create.lifecycle]
        self.assertListEqual(expected, self.bucket.lifecycle_rules)

    def test_patch_called_on_lookup(self):
        create = GcsBucket(self.gcs, 'bucket', exists_ok=True)
        _ = create()
        self.bucket.patch.assert_called_once_with()

    def test_patch_called_on_create(self):
        self.client.lookup_bucket.return_value = None
        create = GcsBucket(self.gcs, 'bucket')
        _ = create()
        self.bucket.patch.assert_called_once_with()

    def test_allowed_kwargs_set(self):
        self.client.lookup_bucket.return_value = None
        create = GcsBucket(self.gcs, 'bucket', foo='bar', baz='cheese')
        self.assertNotEqual('bar', self.bucket.foo)
        self.assertNotEqual('cheese', self.bucket.baz)
        _ = create()
        self.assertEqual('bar', self.bucket.foo)
        self.assertEqual('cheese', self.bucket.baz)

    def test_wrong_kwargs_raise(self):
        self.client.lookup_bucket.return_value = None
        create = GcsBucket(self.gcs, 'bucket', wrong='kwarg')
        with self.assertRaises(GcsError):
            _ = create()



class TestMisc(unittest.TestCase):

    def setUp(self):
        self.project = 'project'
        self.gcs = Mock()
        self.gcs.project = self.project

    def test_default_repr(self):
        create = GcsBucket(self.gcs, 'bucket')
        expected = ("GcsBucket(Mock(...), 'bucket', 'EUROPE-NORTH1',"
                    " False, None, 'project', False)")
        self.assertEqual(expected, repr(create))

    def test_custom_repr(self):
        create = GcsBucket(
            self.gcs,
            'bucket',
            'LOCATION',
            True,
            42,
            'user_project',
            True
        )
        expected = ("GcsBucket(Mock(...), 'bucket', 'LOCATION',"
                    " True, 42, 'user_project', True)")
        self.assertEqual(expected, repr(create))

    def test_pickle_works(self):
        gcs = Gcs('project')
        create = GcsBucket(gcs, 'bucket')
        _ = pickle.loads(pickle.dumps(create))


if __name__ == '__main__':
    unittest.main()
