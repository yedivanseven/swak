import unittest
import pickle
from unittest.mock import patch
from swak.cloud.gcp import Gcs


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.project = 'project'
        self.gcs = Gcs(self.project)

    def test_has_project(self):
        self.assertTrue(hasattr(self.gcs, 'project'))

    def test_project(self):
        self.assertEqual(self.project, self.gcs.project)

    def test_project_stripped(self):
        gcs = Gcs(' ./ project ./ ')
        self.assertEqual(self.project, gcs.project)

    def test_has_args(self):
        self.assertTrue(hasattr(self.gcs, 'args'))

    def test_args(self):
        self.assertTupleEqual((), self.gcs.args)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.gcs, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.gcs.kwargs)

    @patch('swak.cloud.gcp.clients.Client')
    def test_has_client(self, _):
        self.assertTrue(hasattr(self.gcs, 'client'))


class TestAttributes(unittest.TestCase):

    def test_args(self):
        gcs = Gcs('project', 'foo', 42)
        self.assertTupleEqual(('foo', 42), gcs.args)

    def test_kwargs(self):
        gcs = Gcs('project', foo='bar', answer=42)
        self.assertDictEqual({'foo': 'bar', 'answer': 42}, gcs.kwargs)

    def test_args_and_kwargs(self):
        gcs = Gcs('project', 'foo', 42, foo='bar', answer=42)
        self.assertTupleEqual(('foo', 42), gcs.args)
        self.assertDictEqual({'foo': 'bar', 'answer': 42}, gcs.kwargs)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        gcs = Gcs('project')
        self.assertTrue(callable(gcs))

    @patch('swak.cloud.gcp.clients.Client')
    def test_client_called_on_call(self, client):
        gcs = Gcs('project')
        _ = gcs()
        client.assert_called_once_with('project')

    @patch('swak.cloud.gcp.clients.Client')
    def test_client_called_on_call_with_args_kwargs(self, client):
        gcs = Gcs('project', 'foo', 42, foo='bar', answer=42)
        _ = gcs()
        client.assert_called_once_with(
            'project',
            'foo',
            42,
            foo='bar',
            answer=42
        )

    @patch('swak.cloud.gcp.clients.Client')
    def test_args_kwargs_ignored_on_call(self, client):
        gcs = Gcs('project', 'foo', 42, foo='bar', answer=42)
        _ = gcs('bar', 123, baz='foo', reply=123)
        client.assert_called_once_with(
            'project',
            'foo',
            42,
            foo='bar',
            answer=42
        )

    @patch('swak.cloud.gcp.clients.Client')
    def test_client_called_on_each_call(self, client):
        gcs = Gcs('project')
        _ = gcs()
        _ = gcs()
        _ = gcs()
        self.assertEqual(3, client.call_count)

    @patch('swak.cloud.gcp.clients.Client')
    def test_client_called_on_property(self, client):
        gcs = Gcs('project')
        _ = gcs.client
        client.assert_called_once_with('project')

    @patch('swak.cloud.gcp.clients.Client')
    def test_client_called_on_property_with_args_kwargs(self, client):
        gcs = Gcs('project', 'foo', 42, foo='bar', answer=42)
        _ = gcs.client
        client.assert_called_once_with(
            'project',
            'foo',
            42,
            foo='bar',
            answer=42
        )

    @patch('swak.cloud.gcp.clients.Client')
    def test_client_called_once_with_property(self, client):
        gcs = Gcs('project')
        _ = gcs.client
        _ = gcs.client
        _ = gcs.client
        client.assert_called_once_with('project')


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        gcs = Gcs('project')
        self.assertEqual("Gcs('project')", repr(gcs))

    def test_custom_repr(self):
        gcs = Gcs('project', 'foo', 42, foo='bar', answer=42)
        expected = "Gcs('project', 'foo', 42, foo='bar', answer=42)"
        self.assertEqual(expected, repr(gcs))

    def test_pickle_works(self):
        gcs = Gcs('project', 'foo', 42, foo='bar', answer=42)
        _ = pickle.loads(pickle.dumps(gcs))


if __name__ == '__main__':
    unittest.main()
