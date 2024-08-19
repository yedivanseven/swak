import unittest
from unittest.mock import patch, Mock
import pickle
from swak.cloud.gcp import DataFrame2Gbq
from swak.cloud.gcp.exceptions import GbqError


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.write = DataFrame2Gbq('project', 'dataset')

    def test_has_project(self):
        self.assertTrue(hasattr(self.write, 'project'))

    def test_project(self):
        self.assertEqual('project', self.write.project)

    def test_project_strips(self):
        write = DataFrame2Gbq(' /.project ./', 'dataset')
        self.assertEqual('project', write.project)

    def test_has_dataset(self):
        self.assertTrue(hasattr(self.write, 'dataset'))

    def test_dataset(self):
        self.assertEqual('dataset', self.write.dataset)

    def test_dataset_strips(self):
        write = DataFrame2Gbq('project', ' . / dataset /.')
        self.assertEqual('project', write.project)

    def test_has_table(self):
        self.assertTrue(hasattr(self.write, 'table'))

    def test_table(self):
        self.assertEqual('', self.write.table)

    def test_has_location(self):
        self.assertTrue(hasattr(self.write, 'location'))

    def test_location(self):
        self.assertEqual('', self.write.location)

    def test_has_if_exists(self):
        self.assertTrue(hasattr(self.write, 'if_exists'))

    def test_if_exists(self):
        self.assertEqual('fail', self.write.if_exists)

    def test_has_chunksize(self):
        self.assertTrue(hasattr(self.write, 'chunksize'))

    def test_chunksize(self):
        self.assertIsNone(self.write.chunksize)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.write, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.write.kwargs)


class TestAttributes(unittest.TestCase):

    def test_table_strips(self):
        write = DataFrame2Gbq('project', 'dataset', ' ./table /.')
        self.assertEqual('table', write.table)

    def test_location_strips(self):
        write = DataFrame2Gbq('project', 'dataset', 'table', 'location')
        self.assertEqual('location', write.location)

    def test_if_exists(self):
        write = DataFrame2Gbq(
            'project',
            'dataset',
            'table',
            'location',
            'append'
        )
        self.assertEqual('append', write.if_exists)

    def test_if_exists_strips(self):
        write = DataFrame2Gbq(
            'project',
            'dataset',
            'table',
            'location',
            ' aPPenD '
        )
        self.assertEqual('append', write.if_exists)

    def test_chunksize(self):
        write = DataFrame2Gbq(
            'project',
            'dataset',
            'table',
            'location',
            'append',
            42
        )
        self.assertIsInstance(write.chunksize, int)
        self.assertEqual(42, write.chunksize)

    def test_kwargs(self):
        write = DataFrame2Gbq('project', 'dataset', foo='bar')
        self.assertDictEqual({'foo': 'bar'}, write.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.write = DataFrame2Gbq(
            'project',
            'dataset',
            'table',
            'location',
            'if_exists',
            42,
            foo='bar'
        )
        self.df = Mock()
        self.df.reset_index = Mock(return_value='df')

    def test_callable(self):
        self.assertTrue(callable(self.write))

    @patch('pandas_gbq.to_gbq')
    def test_method_called(self, mock):
        _ = self.write(self.df)
        mock.assert_called_once()

    @patch('pandas_gbq.to_gbq')
    def test_method_called_with_instantiation_table(self, mock):
        _ = self.write(self.df)
        mock.assert_called_once_with(
            'df',
            project_id='project',
            destination_table='dataset.table',
            location='location',
            if_exists='if_exists',
            chunksize=42,
            progress_bar=False,
            foo='bar'
        )

    @patch('pandas_gbq.to_gbq')
    def test_method_called_with_call_table(self, mock):
        write = DataFrame2Gbq('project', 'dataset')
        _ = write(self.df, 'call')
        mock.assert_called_once_with(
            'df',
            project_id='project',
            destination_table='dataset.call',
            location=None,
            if_exists='fail',
            chunksize=None,
            progress_bar=False
        )

    @patch('pandas_gbq.to_gbq')
    def test_method_called_with_both_tables(self, mock):
        _ = self.write(self.df, 'call')
        mock.assert_called_once_with(
            'df',
            project_id='project',
            destination_table='dataset.tablecall',
            location='location',
            if_exists='if_exists',
            chunksize=42,
            progress_bar=False,
            foo='bar'
        )

    @patch('pandas_gbq.to_gbq')
    def test_raises_no_table(self, _):
        write = DataFrame2Gbq('project', 'dataset')
        with self.assertRaises(GbqError):
            _ = write(self.df)

    @patch('pandas_gbq.to_gbq')
    def test_function_returns(self, _):
        actual = self.write(self.df)
        self.assertTupleEqual((), actual)


class TestMisc(unittest.TestCase):

    def test_repr(self):
        write = DataFrame2Gbq('project', 'dataset')
        expected = "DataFrame2Gbq('project', 'dataset', '', '', 'fail', None)"
        self.assertEqual(expected, repr(write))

    def test_repr_kwargs(self):
        reader = DataFrame2Gbq(
            'project',
            'dataset',
            'table',
            'location',
            'if_exists',
            42,
            foo='bar'
        )
        expected = ("DataFrame2Gbq('project', 'dataset', 'table', 'location', "
                    "'if_exists', 42, foo='bar')")
        self.assertEqual(expected, repr(reader))

    def test_pickle_works(self):
        reader = DataFrame2Gbq('project', 'dataset')
        _ = pickle.dumps(reader)


if __name__ == '__main__':
    unittest.main()
