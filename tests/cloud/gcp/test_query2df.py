import unittest
from unittest.mock import patch
import pickle
import pandas as pd
from swak.cloud.gcp import GbqQuery2DataFrame


class TestAttributes(unittest.TestCase):

    def test_has_project(self):
        query = GbqQuery2DataFrame('project')
        self.assertTrue(hasattr(query, 'project'))

    def test_project(self):
        query = GbqQuery2DataFrame('project')
        self.assertEqual('project', query.project)

    def test_project_strips(self):
        query = GbqQuery2DataFrame(' /.project ./')
        self.assertEqual('project', query.project)

    def test_has_kwargs(self):
        query = GbqQuery2DataFrame('project')
        self.assertTrue(hasattr(query, 'kwargs'))

    def test_kwargs(self):
        query = GbqQuery2DataFrame('project')
        self.assertDictEqual({}, query.kwargs)

    def test_custom_kwargs(self):
        query = GbqQuery2DataFrame('project', foo='bar')
        self.assertDictEqual({'foo': 'bar'}, query.kwargs)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        query = GbqQuery2DataFrame('project')
        self.assertTrue(callable(query))

    @patch('pandas_gbq.read_gbq')
    def test_function_called(self, mock):
        query = GbqQuery2DataFrame('project')
        _ = query('hello world')
        mock.assert_called_once()

    @patch('pandas_gbq.read_gbq')
    def test_function_called_with_query(self, mock):
        query = GbqQuery2DataFrame('project')
        _ = query('hello world')
        mock.assert_called_once_with(
            'hello world',
            project_id='project',
            progress_bar_type=None
        )

    @patch('pandas_gbq.read_gbq')
    def test_function_called_with_location(self, mock):
        query = GbqQuery2DataFrame('project')
        _ = query('hello world')
        mock.assert_called_once_with(
            'hello world',
            project_id='project',
            progress_bar_type=None
        )

    @patch('pandas_gbq.read_gbq')
    def test_function_called_kwargs(self, method):
        query = GbqQuery2DataFrame('project', foo='bar')
        _ = query('hello world')
        method.assert_called_once_with(
            'hello world',
            project_id='project',
            progress_bar_type=None,
            foo='bar'
        )

    @patch('pandas_gbq.read_gbq', return_value=pd.DataFrame([1, 2, 3]))
    def test_function_returns(self, _):
        query = GbqQuery2DataFrame('project')
        result = query('hello world')
        pd.testing.assert_frame_equal(pd.DataFrame([1, 2, 3]), result)


class TestMisc(unittest.TestCase):

    def test_repr(self):
        reader = GbqQuery2DataFrame('project')
        expected = "GbqQuery2DataFrame('project')"
        self.assertEqual(expected, repr(reader))

    def test_repr_kwargs(self):
        reader = GbqQuery2DataFrame('project', a=3.0, b='foo')
        expected = "GbqQuery2DataFrame('project', a=3.0, b='foo')"
        self.assertEqual(expected, repr(reader))

    def test_pickle_works(self):
        reader = GbqQuery2DataFrame('project')
        _ = pickle.loads(pickle.dumps(reader))


if __name__ == '__main__':
    unittest.main()
