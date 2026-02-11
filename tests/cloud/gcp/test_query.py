import unittest
from unittest.mock import patch, Mock
import pickle
from google.cloud.bigquery import QueryJobConfig
from swak.cloud.gcp import GbqQuery, Gbq
from swak.cloud.gcp.exceptions import GbqError


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        client = Mock()
        self.gbq = Mock(return_value=client)
        self.query = GbqQuery(self.gbq)

    def test_has_gbq(self):
        self.assertTrue(hasattr(self.query, 'gbq'))

    def test_gbq(self):
        self.assertIs(self.query.gbq, self.gbq)

    def test_has_config(self):
        self.assertTrue(hasattr(self.query, 'config'))

    def test_config(self):
        self.assertIsNone(self.query.config)

    def test_has_polling_interval(self):
        self.assertTrue(hasattr(self.query, 'polling_interval'))

    def test_polling_interval(self):
        self.assertIsInstance(self.query.polling_interval, float)
        self.assertEqual(5.0, self.query.polling_interval)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        client = Mock()
        self.gbq = Mock(return_value=client)

    def test_config_works_with_explicit_none(self):
        query = GbqQuery(self.gbq, config=None)
        self.assertIsNone(query.config)

    def test_config_works(self):
        config = QueryJobConfig()
        query = GbqQuery(self.gbq, config=config)
        self.assertIs(query.config, config)

    def test_polling_interval(self):
        query = GbqQuery(self.gbq, polling_interval=3)
        self.assertIsInstance(query.polling_interval, float)
        self.assertEqual(3.0, query.polling_interval)

    def test_polling_interval_raises_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = GbqQuery(self.gbq, polling_interval='foo')

    def test_polling_interval_raises_wrong_value(self):
        with self.assertRaises(ValueError):
            _ = GbqQuery(self.gbq, polling_interval=-3)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.job = Mock()
        self.job.running = Mock(side_effect=[True, False])
        self.job.error_result = None
        self.client = Mock()
        self.client.query = Mock(return_value=self.job)
        self.gbq = Mock(return_value=self.client)

    def test_callable(self):
        query = GbqQuery(self.gbq)
        self.assertTrue(callable(query))

    def test_client_created(self):
        query = GbqQuery(self.gbq, polling_interval=1)
        _ = query('query')
        self.gbq.assert_called_once_with()

    def test_query_called(self):
        query = GbqQuery(
            self.gbq,
            config='config',
            polling_interval=1
        )
        _ = query('query')
        self.client.query.assert_called_once_with('query', 'config')

    @patch('swak.cloud.gcp.query.time.sleep')
    def test_polling_called_correct_number_of_times(self, mock_sleep):
        query = GbqQuery(self.gbq, polling_interval=1)
        self.job.running.side_effect = [True, True, True, False]
        _ = query('query')
        self.assertEqual(3, mock_sleep.call_count)
        mock_sleep.assert_called_with(1)

    def test_query_error(self):
        query = GbqQuery(self.gbq, polling_interval=1)
        self.job.error_result = {'reason': 'error', 'message': 'message'}
        with self.assertRaises(GbqError) as error:
            _ = query('query')
        expected = '\nERROR: message'
        actual = str(error.exception)
        self.assertEqual(expected, actual)

    def test_return_value(self):
        query = GbqQuery(self.gbq, polling_interval=1)
        result = query('query')
        self.assertTupleEqual((), result)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.gbq = Gbq('project')

    def test_repr(self):
        query = GbqQuery(self.gbq)
        expected = "GbqQuery(Gbq('project'), None, 5.0)"
        self.assertEqual(expected, repr(query))

    def test_custom_repr(self):
        query = GbqQuery(
            self.gbq,
            None,
            3
        )
        expected = "GbqQuery(Gbq('project'), None, 3.0)"
        self.assertEqual(expected, repr(query))

    def test_pickle_works(self):
        query = GbqQuery(
            self.gbq,
            None,
            3
        )
        _ = pickle.loads(pickle.dumps(query))


if __name__ == '__main__':
    unittest.main()
