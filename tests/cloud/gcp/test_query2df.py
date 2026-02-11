import unittest
from unittest.mock import patch, Mock
import pickle
import pandas as pd
import polars as pl
import pyarrow as pa
from google.cloud.bigquery import QueryJobConfig
from swak.cloud.gcp import GbqQuery2DataFrame, Gbq
from swak.cloud.gcp.exceptions import GbqError
from swak.io.types import Bears


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        client = Mock()
        self.gbq = Mock(return_value=client)
        self.query = GbqQuery2DataFrame(self.gbq)

    def test_has_gbq(self):
        self.assertTrue(hasattr(self.query, 'gbq'))

    def test_gbq(self):
        self.assertIs(self.query.gbq, self.gbq)

    def test_has_bears(self):
        self.assertTrue(hasattr(self.query, 'bears'))

    def test_bears(self):
        self.assertEqual('pandas', self.query.bears)

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

    def test_bears_works_with_str(self):
        query = GbqQuery2DataFrame(self.gbq, 'pandas')
        self.assertEqual('pandas', query.bears)
        query = GbqQuery2DataFrame(self.gbq, 'polars')
        self.assertEqual('polars', query.bears)

    def test_bears_works_with_enum(self):
        query = GbqQuery2DataFrame(self.gbq, Bears.PANDAS)
        self.assertEqual('pandas', query.bears)
        query = GbqQuery2DataFrame(self.gbq, Bears.POLARS)
        self.assertEqual('polars', query.bears)

    def test_bears_raises_wrong_type(self):
        with self.assertRaises(ValueError):
            _ = GbqQuery2DataFrame(self.gbq, 42)

    def test_bears_raises_wrong_value(self):
        with self.assertRaises(ValueError):
            _ = GbqQuery2DataFrame(self.gbq, 'brown')

    def test_config_works_with_explicit_none(self):
        query = GbqQuery2DataFrame(self.gbq, config=None)
        self.assertIsNone(query.config)

    def test_config_works(self):
        config = QueryJobConfig()
        query = GbqQuery2DataFrame(self.gbq, config=config)
        self.assertIs(query.config, config)

    def test_polling_interval(self):
        query = GbqQuery2DataFrame(self.gbq, polling_interval=3)
        self.assertIsInstance(query.polling_interval, float)
        self.assertEqual(3.0, query.polling_interval)

    def test_polling_interval_raises_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = GbqQuery2DataFrame(self.gbq, polling_interval='foo')

    def test_polling_interval_raises_wrong_value(self):
        with self.assertRaises(ValueError):
            _ = GbqQuery2DataFrame(self.gbq, polling_interval=-3)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.table = pa.table({
            'id': [1, 2, 3],
            'value': ['a', 'b', 'c']
        })
        self.pd = self.table.to_pandas()
        self.pl = pl.from_arrow(self.table)
        self.rows = Mock()
        self.rows.to_arrow = Mock(return_value=self.table)
        self.result = Mock(return_value=self.rows)
        self.job = Mock()
        self.job.running = Mock(side_effect=[True, False])
        self.job.error_result = None
        self.job.result = Mock(return_value=self.rows)
        self.client = Mock()
        self.client.query = Mock(return_value=self.job)
        self.gbq = Mock(return_value=self.client)

    def test_callable(self):
        query = GbqQuery2DataFrame(self.gbq)
        self.assertTrue(callable(query))

    def test_client_created(self):
        query = GbqQuery2DataFrame(self.gbq, polling_interval=1)
        _ = query('query')
        self.gbq.assert_called_once_with()

    def test_query_called(self):
        query = GbqQuery2DataFrame(
            self.gbq,
            config='config',
            polling_interval=1
        )
        _ = query('query')
        self.client.query.assert_called_once_with('query', 'config')

    @patch('swak.cloud.gcp.query2df.time.sleep')
    def test_polling_called_correct_number_of_times(self, mock_sleep):
        query = GbqQuery2DataFrame(self.gbq, polling_interval=1)
        self.job.running.side_effect = [True, True, True, False]
        _ = query('query')
        self.assertEqual(3, mock_sleep.call_count)
        mock_sleep.assert_called_with(1)

    def test_query_error(self):
        query = GbqQuery2DataFrame(self.gbq, polling_interval=1)
        self.job.error_result = {'reason': 'error', 'message': 'message'}
        with self.assertRaises(GbqError) as error:
            _ = query('query')
        expected = '\nERROR: message'
        actual = str(error.exception)
        self.assertEqual(expected, actual)

    def test_return_type_default(self):
        query = GbqQuery2DataFrame(self.gbq, polling_interval=1)
        df = query('query')
        self.assertIsInstance(df, pd.DataFrame)

    def test_return_type_pandas(self):
        query = GbqQuery2DataFrame(self.gbq, 'pandas', polling_interval=1)
        df = query('query')
        self.assertIsInstance(df, pd.DataFrame)

    def test_return_value_pandas(self):
        query = GbqQuery2DataFrame(self.gbq, 'pandas', polling_interval=1)
        df = query('query')
        pd.testing.assert_frame_equal(df, self.pd)

    def test_return_type_polars(self):
        query = GbqQuery2DataFrame(self.gbq, 'polars', polling_interval=1)
        df = query('query')
        self.assertIsInstance(df, pl.DataFrame)

    def test_return_value_polars(self):
        query = GbqQuery2DataFrame(self.gbq, 'polars', polling_interval=1)
        df = query('query')
        self.assertTrue(self.pl.equals(df))


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.gbq = Gbq('project')

    def test_repr(self):
        query = GbqQuery2DataFrame(self.gbq)
        expected = "GbqQuery2DataFrame(Gbq('project'), 'pandas', None, 5.0)"
        self.assertEqual(expected, repr(query))

    def test_custom_repr(self):
        query = GbqQuery2DataFrame(
            self.gbq,
            'polars',
            None,
            3
        )
        expected = "GbqQuery2DataFrame(Gbq('project'), 'polars', None, 3.0)"
        self.assertEqual(expected, repr(query))

    def test_pickle_works(self):
        query = GbqQuery2DataFrame(
            self.gbq,
            'polars',
            None,
            3
        )
        _ = pickle.loads(pickle.dumps(query))


if __name__ == '__main__':
    unittest.main()
