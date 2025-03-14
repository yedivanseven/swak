import pickle
import unittest
from unittest.mock import Mock
import pandas as pd
from swak.pd import SortValues


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.by = 'foo'
        self.sort = SortValues(self.by)

    def test_has_by(self):
        self.assertTrue(hasattr(self.sort, 'by'))

    def test_by(self):
        self.assertEqual(self.by, self.sort.by)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.sort, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.sort.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.by = ['foo', 'bar']
        self.kwargs = {
            'answer': 42,
            'inplace': True
        }
        self.sort = SortValues(self.by, **self.kwargs)

    def test_by(self):
        self.assertEqual(self.by, self.sort.by)

    def test_kwargs(self):
        self.assertDictEqual({'answer': 42}, self.sort.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.by = ['foo', 'bar']
        self.kwargs = {
            'axis': 1,
            'ascending': False
        }
        self.sort = SortValues(self.by, **self.kwargs)


    def test_callable(self):
        self.assertTrue(callable(self.sort))

    def test_sort_values_called(self):
        df = Mock()
        _ = self.sort(df)
        df.sort_values.assert_called_once_with(
            self.by,
            inplace=False,
            **self.kwargs
        )

    def test_return_value(self):
        df = pd.DataFrame([[1, 2, 3, 3], [7, 6, 5, 4]], index=self.by)
        actual = self.sort(df)
        expected = df.sort_values(self.by, **self.kwargs)
        pd.testing.assert_frame_equal(actual, expected)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        sort = SortValues('foo')
        expected = "SortValues('foo')"
        self.assertEqual(expected, repr(sort))

    def test_custom_repr(self):
        sort = SortValues( ['foo', 'bar'], answer=42, inplace=True)
        expected = "SortValues(['foo', 'bar'], answer=42, inplace=True)"
        self.assertEqual(expected, repr(sort))

    def test_pickle_works(self):
        sort = SortValues(['foo', 'bar'], answer=42, inplace=True)
        _ = pickle.loads(pickle.dumps(sort))


if __name__ == '__main__':
    unittest.main()
