import pickle
import unittest
from unittest.mock import Mock
import polars as pl
from swak.pl import GroupByAgg


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.agg = GroupByAgg()

    def test_has_aggs(self):
        self.assertTrue(hasattr(self.agg, 'aggs'))

    def test_aggs(self):
        self.assertTupleEqual((), self.agg.aggs)

    def test_has_named_aggs(self):
        self.assertTrue(hasattr(self.agg, 'named_aggs'))

    def test_named_aggs(self):
        self.assertDictEqual({}, self.agg.named_aggs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.aggs = 'foo', 'bar'
        self.named_aggs = {'baz': 1, 'answer': 42}
        self.agg = GroupByAgg(*self.aggs, **self.named_aggs)

    def test_aggs(self):
        self.assertTupleEqual(self.aggs, self.agg.aggs)

    def test_named_aggs(self):
        self.assertDictEqual(self.named_aggs, self.agg.named_aggs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.aggs = 'foo', 'bar'
        self.named_aggs = {'baz': 1}
        self.agg = GroupByAgg(*self.aggs, **self.named_aggs)

    def test_callable(self):
        self.assertTrue(callable(self.agg))

    def test_agg_called(self):
        df = Mock()
        _ = self.agg(df)
        df.agg.assert_called_once_with(*self.aggs, **self.named_aggs)

    def test_return_value(self):
        df = Mock()
        df.agg = Mock(return_value='answer')
        actual = self.agg(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        agg = GroupByAgg()
        expected = 'GroupByAgg()'
        self.assertEqual(expected, repr(agg))

    def test_custom_repr(self):
        agg = GroupByAgg('foo', 'bar', answer=42)
        expected = "GroupByAgg('foo', 'bar', answer=42)"
        self.assertEqual(expected, repr(agg))

    def test_pickle_works(self):
        agg = GroupByAgg(pl.col('col1').mean(), col=pl.col('col2').max())
        _ = pickle.loads(pickle.dumps(agg))


if __name__ == '__main__':
    unittest.main()
