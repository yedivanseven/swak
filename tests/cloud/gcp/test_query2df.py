import unittest
from unittest.mock import patch
import pickle
import pandas as pd
from swak.cloud.gcp import GbqQuery2DataFrame


class TestAttributes(unittest.TestCase):

    def test_has_project(self):
        query = GbqQuery2DataFrame('project', 'location')
        self.assertTrue(hasattr(query, 'project'))

    def test_project(self):
        query = GbqQuery2DataFrame('project', 'location')
        self.assertEqual('project', query.project)

    def test_project_strips(self):
        query = GbqQuery2DataFrame(' /.project ./', 'location')
        self.assertEqual('project', query.project)

    def test_has_location(self):
        query = GbqQuery2DataFrame('project', 'location')
        self.assertTrue(hasattr(query, 'location'))

    def test_location(self):
        query = GbqQuery2DataFrame('project', 'location')
        self.assertEqual('location', query.location)

    def test_location_strips(self):
        query = GbqQuery2DataFrame('project', '  loCaTIon ')
        self.assertEqual('location', query.location)

    def test_has_kwargs(self):
        query = GbqQuery2DataFrame('project', 'location')
        self.assertTrue(hasattr(query, 'kwargs'))

    def test_kwargs(self):
        query = GbqQuery2DataFrame('project', 'location')
        self.assertDictEqual({}, query.kwargs)

    def test_custom_kwargs(self):
        query = GbqQuery2DataFrame('project', 'location', foo='bar')
        self.assertDictEqual({'foo': 'bar'}, query.kwargs)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        query = GbqQuery2DataFrame('project', 'location')
        self.assertTrue(callable(query))

    @patch('pandas_gbq.read_gbq')
    def test_function_called(self, mock):
        query = GbqQuery2DataFrame('project', 'location')
        _ = query('hello world')
        mock.assert_called_once()

    @patch('pandas_gbq.read_gbq')
    def test_function_called_with_query(self, mock):
        query = GbqQuery2DataFrame('project', 'location')
        _ = query('hello world')
        mock.assert_called_once_with(
            'hello world',
            project_id='project',
            location='location',
            progress_bar_type=None
        )

    @patch('pandas_gbq.read_gbq')
    def test_function_called_with_interpolated_query(self, mock):
        query = GbqQuery2DataFrame('project', 'location')
        _ = query('hello {}', 'world')
        mock.assert_called_once_with(
            'hello world',
            project_id='project',
            location='location',
            progress_bar_type=None
        )

    @patch('pandas_gbq.read_gbq')
    def test_function_called_kwargs(self, method):
        query = GbqQuery2DataFrame('project', 'location', foo='bar')
        _ = query('hello world')
        method.assert_called_once_with(
            'hello world',
            project_id='project',
            location='location',
            progress_bar_type=None,
            foo='bar'
        )

    @patch('pandas_gbq.read_gbq', return_value=pd.DataFrame([1, 2, 3]))
    def test_function_returns(self, _):
        query = GbqQuery2DataFrame('project', 'location')
        result = query('hello world')
        pd.testing.assert_frame_equal(pd.DataFrame([1, 2, 3]), result)


class TestMisc(unittest.TestCase):

    def test_repr(self):
        reader = GbqQuery2DataFrame('project', 'location')
        expected = "GbqQuery2DataFrame('project', 'location')"
        self.assertEqual(expected, repr(reader))

    def test_repr_kwargs(self):
        reader = GbqQuery2DataFrame('project', 'location', a=3.0, b='foo')
        expected = "GbqQuery2DataFrame('project', 'location', a=3.0, b='foo')"
        self.assertEqual(expected, repr(reader))

    def test_pickle_works(self):
        reader = GbqQuery2DataFrame('project', 'location')
        _ = pickle.dumps(reader)


if __name__ == '__main__':
    unittest.main()
