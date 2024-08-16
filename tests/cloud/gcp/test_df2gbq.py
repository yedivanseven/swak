import unittest
from unittest.mock import patch, Mock
import pickle
from swak.cloud.gcp import DataFrame2Gbq


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.write = DataFrame2Gbq('project', 'location', 'table')

    def test_has_project(self):
        self.assertTrue(hasattr(self.write, 'project'))

    def test_project(self):
        self.assertEqual('project', self.write.project)

    def test_project_strips(self):
        write = DataFrame2Gbq(' /.project ./', 'location', 'table')
        self.assertEqual('project', write.project)

    def test_has_location(self):
        self.assertTrue(hasattr(self.write, 'location'))

    def test_location(self):
        self.assertEqual('location', self.write.location)

    def test_location_strips(self):
        write = DataFrame2Gbq('project', '  loCaTIon ', 'table')
        self.assertEqual('location', write.location)

    def test_has_table(self):
        self.assertTrue(hasattr(self.write, 'table'))

    def test_table(self):
        self.assertEqual('table', self.write.table)

    def test_table_strips(self):
        write = DataFrame2Gbq('project', 'location', ' ./table /.')
        self.assertEqual('table', write.table)

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

    def setUp(self):
        self.write = DataFrame2Gbq(
            'project',
            'location',
            'table',
            'if_exists',
            42,
            foo='bar'
        )
        self.df = Mock()
        self.df.reset_index = Mock()

    def test_if_exists(self):
        self.assertEqual('if_exists', self.write.if_exists)

    def test_if_exists_strips(self):
        write = DataFrame2Gbq(
            'project',
            'location',
            'table',
            ' iF_EXists  '
        )
        self.assertEqual('if_exists', write.if_exists)

    def test_chunksize(self):
        self.assertIsInstance(self.write.chunksize, int)
        self.assertEqual(42, self.write.chunksize)

    def test_kwargs(self):
        self.assertDictEqual({'foo': 'bar'}, self.write.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.write = DataFrame2Gbq(
            'project',
            'location',
            'table',
            'if_exists',
            42,
            foo='bar'
        )
        self.df = Mock()
        self.df.reset_index = Mock(return_value='df')

    def test_callable(self):
        self.assertTrue(callable(self.write))

    @patch('pandas_gbq.to_gbq')
    def test_function_called(self, mock):
        _ = self.write(self.df)
        mock.assert_called_once()

    @patch('pandas_gbq.to_gbq')
    def test_method_function_with_kwargs(self, mock):
        _ = self.write(self.df)
        mock.assert_called_once_with(
            'df',
            project_id='project',
            location='location',
            destination_table='table',
            if_exists='if_exists',
            chunksize=42,
            progress_bar=False,
            foo='bar'
        )

    @patch('pandas_gbq.to_gbq')
    def test_function_returns(self, _):
        actual = self.write(self.df)
        self.assertTupleEqual((), actual)


class TestMisc(unittest.TestCase):

    def test_repr(self):
        reader = DataFrame2Gbq('project', 'location', 'table')
        expected = "DataFrame2Gbq('project', 'location', 'table', 'fail', None)"
        self.assertEqual(expected, repr(reader))

    def test_repr_kwargs(self):
        reader = DataFrame2Gbq(
            'project',
            'location',
            'table',
            'if_exists',
            42,
            foo='bar'
        )
        expected = ("DataFrame2Gbq('project', 'location', 'table', "
                    "'if_exists', 42, foo='bar')")
        self.assertEqual(expected, repr(reader))

    def test_pickle_works(self):
        reader = DataFrame2Gbq('project', 'location', 'table')
        _ = pickle.dumps(reader)


if __name__ == '__main__':
    unittest.main()
