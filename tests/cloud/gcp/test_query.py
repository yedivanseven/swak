import unittest
import pickle
from unittest.mock import Mock, patch
from swak.cloud.gcp import GbqQuery
from swak.cloud.gcp.exceptions import GbqError


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.query = GbqQuery('project')

    def test_project(self):
        self.assertTrue(hasattr(self.query, 'project'))
        self.assertEqual('project', self.query.project)

    def test_polling_interval(self):
        self.assertTrue(hasattr(self.query, 'polling_interval'))
        self.assertIsInstance(self.query.polling_interval, int)
        self.assertEqual(5, self.query.polling_interval)

    def test_priority(self):
        self.assertTrue(hasattr(self.query, 'priority'))
        self.assertEqual('BATCH', self.query.priority)

    def test_kwargs(self):
        self.assertTrue(hasattr(self.query, 'kwargs'))
        self.assertDictEqual({}, self.query.kwargs)

    def test_project_stripped(self):
        query = GbqQuery(' /.project ./')
        self.assertEqual('project', query.project)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.query = GbqQuery(
            'project',
            7,
            ' INterACTivE  ',
            hello='world'
        )

    def test_polling_interval(self):
        self.assertIsInstance(self.query.polling_interval, int)
        self.assertEqual(7, self.query.polling_interval)

    def test_priority(self):
        self.assertEqual('INTERACTIVE', self.query.priority)

    def test_kwargs(self):
        self.assertDictEqual({'hello': 'world'}, self.query.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.job = Mock()
        self.job.running.return_value = False
        self.job.error_result = {}
        self.client_instance = Mock()
        self.client_instance.query.return_value = self.job
        self.client_patch = patch(
            'swak.cloud.gcp.query.Client',
            return_value=self.client_instance
        )
        self.config_patch = patch(
            'swak.cloud.gcp.query.QueryJobConfig',
            return_value='config'
        )
        self.client_class = self.client_patch.start()
        self.config = self.config_patch.start()
        self.query = GbqQuery(
            'project',
            7,
            'INTERACTIVE',
            hello='world'
        )

    def tearDown(self) -> None:
        self.client_patch.stop()
        self.config_patch.stop()

    def test_callable(self):
        self.assertTrue(callable(self.query))

    def test_client_called_once(self):
        _ = self.query('SELECT 1')
        self.client_class.assert_called_once()

    def test_client_called_with_kwargs(self):
        _ = self.query('SELECT 1', foo='bar')
        self.client_class.assert_called_once_with(
            self.query.project,
            hello='world'
        )

    def test_config_called_once(self):
        _ = self.query('SELECT 1')
        self.config.assert_called_once()

    def test_config_called_with_kwargs(self):
        _ = self.query('SELECT 1', foo='bar')
        self.config.assert_called_once_with(
            priority=self.query.priority,
            foo='bar'
        )

    def test_query_called_once(self):
        _ = self.query('SELECT 1')
        self.client_instance.query.assert_called_once()

    def test_query_called_with_args(self):
        _ = self.query('SELECT 1')
        self.client_instance.query.assert_called_once_with(
            'SELECT 1',
            'config'
        )

    def test_job_running_called_once(self):
        _ = self.query('SELECT 1')
        self.job.running.assert_called_once_with()

    def test_job_raises(self):
        self.job.error_result = {'reason': 'reason', 'message': 'message'}
        with self.assertRaises(GbqError):
            _ = self.query('SELECT 1')

    def test_return_value(self):
        actual = self.query('SELECT 1')
        self.assertTupleEqual((), actual)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.query = GbqQuery(
            'project',
            7,
            'INTERACTIVE',
            hello='world'
        )

    def test_repr(self):
        expected = "GbqQuery('project', 7, 'INTERACTIVE', hello='world')"
        self.assertEqual(expected, repr(self.query))

    def test_pickle_works(self):
        _ = pickle.dumps(self.query)


if __name__ == '__main__':
    unittest.main()
