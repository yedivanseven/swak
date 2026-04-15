import pickle
import unittest
from unittest.mock import Mock
import pandas as pd
from swak.pd import SortValues


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.by = 'foo'
        self.sort = SortValues(self.by)

    def test_has_bys(self):
        self.assertTrue(hasattr(self.sort, 'bys'))

    def test_bys(self):
        self.assertListEqual([self.by], self.sort.bys)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.sort, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.sort.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.by = ['foo', 'bar']
        self.bys = 'baz'
        self.kwargs = {
            'answer': 42,
            'inplace': True
        }
        self.sort = SortValues(self.by, self.bys, **self.kwargs)

    def test_bys(self):
        self.assertEqual([*self.by, self.bys], self.sort.bys)

    def test_kwargs(self):
        self.assertDictEqual({'answer': 42}, self.sort.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.by = ['foo', 'bar']
        self.bys = 'baz'
        self.kwargs = {
            'axis': 1,
            'ascending': False
        }
        self.sort = SortValues(self.by, self.bys, **self.kwargs)


    def test_callable(self):
        self.assertTrue(callable(self.sort))

    def test_dataframe(self):
        df = pd.DataFrame(range(10))
        df.sort_values = Mock(return_value='answer')
        actual = self.sort(df)
        df.sort_values.assert_called_once_with(
            [*self.by, self.bys],
            inplace=False,
            **self.kwargs
        )
        self.assertEqual('answer', actual)

    def test_series(self):
        df = pd.Series(range(10))
        df.sort_values = Mock(return_value='answer')
        actual = self.sort(df)
        df.sort_values.assert_called_once_with(
            inplace=False,
            **self.kwargs
        )
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        sort = SortValues('foo')
        expected = "SortValues(['foo'])"
        self.assertEqual(expected, repr(sort))

    def test_custom_repr(self):
        sort = SortValues( ['foo', 'bar'], answer=42, inplace=True)
        expected = "SortValues(['foo', 'bar'], answer=42)"
        self.assertEqual(expected, repr(sort))

    def test_pickle_works(self):
        sort = SortValues(['foo', 'bar'], answer=42, inplace=True)
        _ = pickle.loads(pickle.dumps(sort))


if __name__ == '__main__':
    unittest.main()
