import pickle
import unittest
from unittest.mock import Mock
import polars as pl
from swak.pl import Filter


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.filter = Filter()

    def test_has_predicates(self):
        self.assertTrue(hasattr(self.filter, 'predicates'))

    def test_predicates(self):
        self.assertTupleEqual((), self.filter.predicates)

    def test_has_named_constraints(self):
        self.assertTrue(hasattr(self.filter, 'constraints'))

    def test_constraints(self):
        self.assertDictEqual({}, self.filter.constraints)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.predicates = 'foo', 'bar'
        self.constraints = {'baz': 1, 'answer': 42}
        self.filter = Filter(*self.predicates, **self.constraints)

    def test_predicates(self):
        self.assertTupleEqual(self.predicates, self.filter.predicates)

    def test_constraints(self):
        self.assertDictEqual(self.constraints, self.filter.constraints)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.predicates = 'foo', 'bar'
        self.constraints = {'baz': 1}
        self.filter = Filter(*self.predicates, **self.constraints)

    def test_callable(self):
        self.assertTrue(callable(self.filter))

    def test_filter_called(self):
        df = Mock()
        _ = self.filter(df)
        df.filter.assert_called_once_with(*self.predicates, **self.constraints)

    def test_return_value(self):
        df = Mock()
        df.filter = Mock(return_value='answer')
        actual = self.filter(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        filter_df = Filter()
        expected = 'Filter()'
        self.assertEqual(expected, repr(filter_df))

    def test_custom_repr(self):
        filter_df = Filter('foo', 'bar', answer=42)
        expected = "Filter('foo', 'bar', answer=42)"
        self.assertEqual(expected, repr(filter_df))

    def test_pickle_works(self):
        filter_df = Filter(pl.col('col1')==2, col2=5)
        _ = pickle.loads(pickle.dumps(filter_df))


if __name__ == '__main__':
    unittest.main()
