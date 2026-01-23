import unittest
import pickle
from unittest.mock import patch
from swak.cloud.gcp import Gbq


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.project = 'project'
        self.gbq = Gbq(self.project)

    def test_has_project(self):
        self.assertTrue(hasattr(self.gbq, 'project'))

    def test_project(self):
        self.assertEqual(self.project, self.gbq.project)

    def test_project_stripped(self):
        gcs = Gbq(' ./ project ./ ')
        self.assertEqual(self.project, gcs.project)

    def test_has_args(self):
        self.assertTrue(hasattr(self.gbq, 'args'))

    def test_args(self):
        self.assertTupleEqual((), self.gbq.args)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.gbq, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.gbq.kwargs)

    @patch('swak.cloud.gcp.clients.GbqClient')
    def test_has_client(self, _):
        self.assertTrue(hasattr(self.gbq, 'client'))


class TestAttributes(unittest.TestCase):

    def test_args(self):
        gbq = Gbq('project', 'foo', 42)
        self.assertTupleEqual(('foo', 42), gbq.args)

    def test_kwargs(self):
        gbq = Gbq('project', foo='bar', answer=42)
        self.assertDictEqual({'foo': 'bar', 'answer': 42}, gbq.kwargs)

    def test_args_and_kwargs(self):
        gbq = Gbq('project', 'foo', 42, foo='bar', answer=42)
        self.assertTupleEqual(('foo', 42), gbq.args)
        self.assertDictEqual({'foo': 'bar', 'answer': 42}, gbq.kwargs)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        gbq = Gbq('project')
        self.assertTrue(callable(gbq))

    @patch('swak.cloud.gcp.clients.GbqClient')
    def test_client_called_on_call(self, client):
        gbq = Gbq('project')
        _ = gbq()
        client.assert_called_once_with('project')

    @patch('swak.cloud.gcp.clients.GbqClient')
    def test_client_called_on_call_with_args_kwargs(self, client):
        gbq = Gbq('project', 'foo', 42, foo='bar', answer=42)
        _ = gbq()
        client.assert_called_once_with(
            'project',
            'foo',
            42,
            foo='bar',
            answer=42
        )

    @patch('swak.cloud.gcp.clients.GbqClient')
    def test_args_kwargs_ignored_on_call(self, client):
        gbq = Gbq('project', 'foo', 42, foo='bar', answer=42)
        _ = gbq('bar', 123, baz='foo', reply=123)
        client.assert_called_once_with(
            'project',
            'foo',
            42,
            foo='bar',
            answer=42
        )

    @patch('swak.cloud.gcp.clients.GbqClient')
    def test_client_called_on_each_call(self, client):
        gbq = Gbq('project')
        _ = gbq()
        _ = gbq()
        _ = gbq()
        self.assertEqual(3, client.call_count)

    @patch('swak.cloud.gcp.clients.GbqClient')
    def test_client_called_on_property(self, client):
        gbq = Gbq('project')
        _ = gbq.client
        client.assert_called_once_with('project')

    @patch('swak.cloud.gcp.clients.GbqClient')
    def test_client_called_on_property_with_args_kwargs(self, client):
        gbq = Gbq('project', 'foo', 42, foo='bar', answer=42)
        _ = gbq.client
        client.assert_called_once_with(
            'project',
            'foo',
            42,
            foo='bar',
            answer=42
        )

    @patch('swak.cloud.gcp.clients.GbqClient')
    def test_client_called_once_with_property(self, client):
        gbq = Gbq('project')
        _ = gbq.client
        _ = gbq.client
        _ = gbq.client
        client.assert_called_once_with('project')


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        gbq = Gbq('project')
        self.assertEqual("Gbq('project')", repr(gbq))

    def test_custom_repr(self):
        gbq = Gbq('project', 'foo', 42, foo='bar', answer=42)
        expected = "Gbq('project', 'foo', 42, foo='bar', answer=42)"
        self.assertEqual(expected, repr(gbq))

    def test_pickle_works(self):
        gbq = Gbq('project', 'foo', 42, foo='bar', answer=42)
        _ = pickle.loads(pickle.dumps(gbq))


if __name__ == '__main__':
    unittest.main()
