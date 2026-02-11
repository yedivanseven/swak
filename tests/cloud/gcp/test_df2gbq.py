import unittest
from unittest.mock import patch, Mock
import pickle
from io import BytesIO
from pandas import DataFrame as Pandas
from polars import DataFrame as Polars
from google.cloud.bigquery import LoadJobConfig, SourceFormat
from google.cloud.exceptions import NotFound
from swak.cloud.gcp import DataFrame2Gbq, ParquetLoadJobConfig, Gbq
from swak.cloud.gcp.exceptions import GbqError


class TestParquetLoadJobConfig(unittest.TestCase):

    def test_empty_has_kwargs(self):
        config = ParquetLoadJobConfig()
        self.assertTrue(hasattr(config, 'kwargs'))

    def test_empty_kwargs(self):
        config = ParquetLoadJobConfig()
        self.assertDictEqual({}, config.kwargs)

    def test_empty_source_format_popped(self):
        config = ParquetLoadJobConfig(source_format='source_format')
        self.assertDictEqual({}, config.kwargs)

    def test_has_kwargs(self):
        config = ParquetLoadJobConfig(foo='bar', baz=42)
        self.assertTrue(hasattr(config, 'kwargs'))

    def test_kwargs(self):
        config = ParquetLoadJobConfig(foo='bar', baz=42)
        self.assertDictEqual({'foo': 'bar', 'baz': 42}, config.kwargs)

    def test_source_format_popped(self):
        config = ParquetLoadJobConfig(foo='bar', baz=42, source_format='fmt')
        self.assertDictEqual({'foo': 'bar', 'baz': 42}, config.kwargs)

    def test_callable(self):
        config = ParquetLoadJobConfig()
        self.assertTrue(callable(config))

    def test_return_type(self):
        config = ParquetLoadJobConfig()
        actual = config()
        self.assertIsInstance(actual, LoadJobConfig)

    @patch('swak.cloud.gcp.df2gbq.LoadJobConfig')
    def test_load_job_config_called_empty(self, mock):
        config = ParquetLoadJobConfig()
        _ = config()
        mock.assert_called_once_with(source_format=SourceFormat.PARQUET)

    @patch('swak.cloud.gcp.df2gbq.LoadJobConfig')
    def test_load_job_config_called_kwargs(self, mock):
        config = ParquetLoadJobConfig(foo='bar', baz=42)
        _ = config()
        mock.assert_called_once_with(
            source_format=SourceFormat.PARQUET,
            foo='bar',
            baz=42
        )

    @patch('swak.cloud.gcp.df2gbq.LoadJobConfig')
    def test_load_job_config_called_kwargs_popped(self, mock):
        config = ParquetLoadJobConfig(foo='bar', baz=42, source_format='fmt')
        _ = config()
        mock.assert_called_once_with(
            source_format=SourceFormat.PARQUET,
            foo='bar',
            baz=42
        )

    @patch('swak.cloud.gcp.df2gbq.LoadJobConfig')
    def test_accepts_args(self, mock):
        config = ParquetLoadJobConfig()
        _ = config('foo', 42)
        mock.assert_called_once_with(source_format=SourceFormat.PARQUET)

    @patch('swak.cloud.gcp.df2gbq.LoadJobConfig')
    def test_accepts_kwargs(self, mock):
        config = ParquetLoadJobConfig()
        _ = config(bar='baz')
        mock.assert_called_once_with(source_format=SourceFormat.PARQUET)

    @patch('swak.cloud.gcp.df2gbq.LoadJobConfig')
    def test_accepts_args_and_kwargs(self, mock):
        config = ParquetLoadJobConfig()
        _ = config('foo', 42, bar='baz')
        mock.assert_called_once_with(source_format=SourceFormat.PARQUET)


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        client = Mock()
        self.gbq = Mock(return_value=client)
        self.dataset = 'dataset'
        self.write = DataFrame2Gbq(self.gbq, self.dataset)

    def test_has_gbq(self):
        self.assertTrue(hasattr(self.write, 'gbq'))

    def test_gbq(self):
        self.assertIs(self.write.gbq, self.gbq)

    def test_has_dataset(self):
        self.assertTrue(hasattr(self.write, 'dataset'))

    def test_dataset(self):
        self.assertEqual(self.dataset, self.write.dataset)

    def test_dataset_strips(self):
        write = DataFrame2Gbq(self.gbq, ' . / dataset /.')
        self.assertEqual(self.dataset, write.dataset)

    def test_dataset_last_part(self):
        write = DataFrame2Gbq(self.gbq, 'project.dataset')
        self.assertEqual(self.dataset, write.dataset)

    def test_non_string_dataset_raises(self):
        with self.assertRaises(TypeError):
            _ = DataFrame2Gbq(self.gbq, 42)

    def test_empty_dataset_raises(self):
        with self.assertRaises(ValueError):
            _ = DataFrame2Gbq(self.gbq, '')

    def test_empty_stripped_dataset_raises(self):
        with self.assertRaises(ValueError):
            _ = DataFrame2Gbq(self.gbq, ' / ./ . ')

    def test_has_table(self):
        self.assertTrue(hasattr(self.write, 'table'))

    def test_table(self):
        self.assertEqual('{}', self.write.table)

    def test_has_location(self):
        self.assertTrue(hasattr(self.write, 'location'))

    def test_location(self):
        self.assertEqual('europe-north1', self.write.location)

    def test_has_config(self):
        self.assertTrue(hasattr(self.write, 'config'))

    def test_config(self):
        self.assertIsInstance(self.write.config, ParquetLoadJobConfig)

    def test_has_polling_interval(self):
        self.assertTrue(hasattr(self.write, 'polling_interval'))

    def test_polling_interval(self):
        self.assertIsInstance(self.write.polling_interval, float)
        self.assertEqual(5, self.write.polling_interval)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.write, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.write.kwargs)

    def test_callable(self):
        self.assertTrue(callable(self.write))


class TestAttributes(unittest.TestCase):

    def setUp(self):
        client = Mock()
        self.gbq = Mock(return_value=client)
        self.dataset = 'dataset'

    def test_table_strips(self):
        write = DataFrame2Gbq(self.gbq, self.dataset, ' . / table ./')
        self.assertEqual('table', write.table)

    def test_table_last_part(self):
        write = DataFrame2Gbq(self.gbq, self.dataset, 'project.dataset.table')
        self.assertEqual('table', write.table)

    def test_non_string_table_raises(self):
        with self.assertRaises(TypeError):
            _ = DataFrame2Gbq(self.gbq, 42)

    def test_empty_table_raises(self):
        with self.assertRaises(ValueError):
            _ = DataFrame2Gbq(self.gbq, self.dataset, '')

    def test_empty_stripped_table_raises(self):
        with self.assertRaises(ValueError):
            _ = DataFrame2Gbq(self.gbq, self.dataset, ' / ./ . ')

    def test_location_strips_lowers(self):
        write = DataFrame2Gbq(self.gbq, self.dataset, 'table', ' LoC ')
        self.assertEqual('loc', write.location)

    def test_location_rises_wrong_type(self):
        with self.assertRaises(AttributeError):
            _ = DataFrame2Gbq(self.gbq, self.dataset, 'table', 42)

    def test_config(self):
        config = ParquetLoadJobConfig()
        write = DataFrame2Gbq(self.gbq, self.dataset, config=config)
        self.assertIs(write.config, config)

    def test_polling_interval(self):
        write = DataFrame2Gbq(self.gbq, self.dataset, polling_interval=32)
        self.assertIsInstance(write.polling_interval, float)
        self.assertEqual(32, write.polling_interval)

    def test_polling_interval_raises_on_uncastable(self):
        with self.assertRaises(TypeError):
            _ = DataFrame2Gbq(self.gbq, self.dataset, polling_interval='foo')

    def test_polling_interval_raises_wrong_number(self):
        with self.assertRaises(ValueError):
            _ = DataFrame2Gbq(self.gbq, self.dataset, polling_interval=0.3)

    def test_kwargs(self):
        write = DataFrame2Gbq(self.gbq, self.dataset, foo='bar', baz=42)
        self.assertDictEqual({'foo': 'bar', 'baz': 42}, write.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.location = 'table_location'
        table = Mock()
        table.location = self.location
        self.job = Mock()
        self.job.running = Mock(side_effect=[True, False])
        self.job.error_result = None
        self.get_table = Mock(return_value=table)
        self.client = Mock()
        self.client.get_table = self.get_table
        self.client.load_table_from_file = Mock(return_value=self.job)
        self.gbq = Mock(return_value=self.client)
        self.dataset = 'dataset'
        self.table = 'table'
        self.pandas = Pandas(range(10))
        self.polars = Polars(range(10))
        self.config = Mock(return_value='config')
        self.write = DataFrame2Gbq(
            self.gbq,
            self.dataset,
            self.table,
            polling_interval=1,
            config=self.config
        )


    def test_client_created(self):
        write = DataFrame2Gbq(
            self.gbq,
            self.dataset,
            polling_interval=1
        )
        _ = write(self.pandas, 'table')
        self.gbq.assert_called_once_with()

    def test_table_interpolated_and_stripped(self):
        write = DataFrame2Gbq(
            self.gbq,
            self.dataset,
            polling_interval=1
        )
        _ = write(self.pandas, ' . / table ./')
        self.get_table.assert_called_once_with('dataset.table')

    def test_table_interpolated_and_last_part(self):
        write = DataFrame2Gbq(
            self.gbq,
            self.dataset,
            polling_interval=1
        )
        _ = write(self.pandas, 'other.table')
        self.get_table.assert_called_once_with('dataset.table')

    def test_get_table_called_with_instantiation_values(self):
        _ = self.write(self.pandas)
        self.get_table.assert_called_once_with('dataset.table')

    def test_pandas_called_empty_kwargs(self):
        df = Mock()
        _ = self.write(df)
        df.to_parquet.assert_called_once()
        stream = df.to_parquet.call_args[0][0]
        self.assertIsInstance(stream, BytesIO)

    def test_pandas_called_kwargs(self):
        write = DataFrame2Gbq(
            self.gbq,
            self.dataset,
            self.table,
            polling_interval=1,
            foo='bar',
            baz=42
        )
        df = Mock()
        _ = write(df)
        df.to_parquet.assert_called_once()
        kwargs = df.to_parquet.call_args[1]
        self.assertDictEqual({'foo': 'bar', 'baz': 42}, kwargs)

    def test_polars_called_empty_kwargs(self):
        df = Mock()
        del df.to_parquet
        df.write_parquet = Mock()
        _ = self.write(df)
        df.write_parquet.assert_called_once()

    def test_polars_called_kwargs(self):
        write = DataFrame2Gbq(
            self.gbq,
            self.dataset,
            self.table,
            polling_interval=1,
            foo='bar',
            baz=42
        )
        df = Mock()
        del df.to_parquet
        df.write_parquet = Mock()
        _ = write(df)
        df.write_parquet.assert_called_once()
        kwargs = df.write_parquet.call_args[1]
        self.assertDictEqual({'foo': 'bar', 'baz': 42}, kwargs)

    def test_load_file_called_table_location(self):
        _ = self.write(self.pandas)
        self.client.load_table_from_file.assert_called_once()
        kwargs = self.client.load_table_from_file.call_args[1]
        self.assertIsInstance(kwargs['file_obj'], BytesIO)
        self.assertEqual(
            self.dataset + '.' + self.table,
            kwargs['destination']
        )
        self.assertIsInstance(kwargs['rewind'], bool)
        self.assertTrue(kwargs['rewind'])
        self.assertEqual(self.location, kwargs['location'])
        self.assertEqual('config', kwargs['job_config'])

    def test_load_file_called_location(self):
        self.get_table.side_effect = NotFound('Table does not exist!')
        _ = self.write(self.pandas)
        self.client.load_table_from_file.assert_called_once()
        kwargs = self.client.load_table_from_file.call_args[1]
        self.assertIsInstance(kwargs['file_obj'], BytesIO)
        self.assertEqual(
            self.dataset + '.' + self.table,
            kwargs['destination']
        )
        self.assertIsInstance(kwargs['rewind'], bool)
        self.assertTrue(kwargs['rewind'])
        self.assertEqual(self.write.location, kwargs['location'])
        self.assertEqual('config', kwargs['job_config'])

    @patch('swak.cloud.gcp.df2gbq.time.sleep')
    def test_polling_called_correct_number_of_times(self, mock_sleep):
        self.job.running.side_effect = [True, True, True, False]
        _ = self.write(self.pandas)
        self.assertEqual(3, mock_sleep.call_count)
        mock_sleep.assert_called_with(1)

    def test_upload_error(self):
        self.job.error_result = {'reason': 'error', 'message': 'message'}
        with self.assertRaises(GbqError) as error:
            _ = self.write(self.pandas)
        expected = '\nERROR: message'
        actual = str(error.exception)
        self.assertEqual(expected, actual)

    def test_return_value_pandas(self):
        actual = self.write(self.pandas)
        self.assertTupleEqual((), actual)

    def test_return_value_polars(self):
        actual = self.write(self.polars)
        self.assertTupleEqual((), actual)


class TestMisc(unittest.TestCase):

    def test_repr(self):
        write = DataFrame2Gbq(Gbq('project'), 'dataset')
        expected = ("DataFrame2Gbq(Gbq('project'), 'dataset', '{}', "
                    "'europe-north1', ParquetLoadJobConfig(), 5.0)")
        self.assertEqual(expected, repr(write))

    def test_repr_kwargs(self):
        write = DataFrame2Gbq(
            Gbq('project'),
            'dataset',
            'table',
            'location',
            'config',
            32,
            foo='bar'
        )
        expected = ("DataFrame2Gbq(Gbq('project'), 'dataset', 'table', "
                    "'location', 'config', 32.0, foo='bar')")
        self.assertEqual(expected, repr(write))

    def test_pickle_works(self):
        write = DataFrame2Gbq(Gbq('project'), 'dataset')
        _ = pickle.loads(pickle.dumps(write))


if __name__ == '__main__':
    unittest.main()
