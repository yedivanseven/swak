import pickle
import unittest
from unittest.mock import Mock
import polars as pl
from swak.pl import DropNulls


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.drop = DropNulls()

    def test_has_subset(self):
        self.assertTrue(hasattr(self.drop, 'subset'))

    def test_subset(self):
        self.assertIsNone(self.drop.subset)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.subset = 'foo', pl.col('bar')
        self.drop = DropNulls(self.subset)

    def test_subset(self):
        self.assertTupleEqual(self.subset, self.drop.subset)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.subset = 'foo', pl.col('bar')
        self.drop = DropNulls(self.subset)

    def test_callable(self):
        self.assertTrue(callable(self.drop))

    def test_drop_called(self):
        df = Mock()
        _ = self.drop(df)
        df.drop_nulls.assert_called_once_with(self.subset)

    def test_return_value(self):
        df = Mock()
        df.drop_nulls = Mock(return_value='answer')
        actual = self.drop(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        drop = DropNulls('foo')
        expected = "DropNulls('foo')"
        self.assertEqual(expected, repr(drop))

    def test_custom_repr(self):
        drop = DropNulls(pl.col('bar'))
        expected = "DropNulls(PolarsExpr)"
        self.assertEqual(expected, repr(drop))

    def test_pickle_works(self):
        drop = DropNulls(pl.col('bar'))
        _ = pickle.loads(pickle.dumps(drop))


if __name__ == '__main__':
    unittest.main()
