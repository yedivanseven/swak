import pickle
import unittest
from unittest.mock import Mock
import polars as pl
from swak.pl import Pivot


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.on = 'foo'
        self.pivot = Pivot(self.on)

    def test_has_on(self):
        self.assertTrue(hasattr(self.pivot, 'on'))

    def test_on(self):
        self.assertEqual(self.on, self.pivot.on)

    def test_has_on_columns(self):
        self.assertTrue(hasattr(self.pivot, 'on_columns'))

    def test_on_columns(self):
        self.assertIsNone(self.pivot.on_columns)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.pivot, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.pivot.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.on = 'foo'
        self.on_columns = 'bar', 'baz'
        self.kwargs = {'answer': 42}
        self.pivot = Pivot(self.on, self.on_columns, **self.kwargs)

    def test_on_columns(self):
        self.assertTupleEqual(self.on_columns, self.pivot.on_columns)

    def test_kwargs(self):
        self.assertDictEqual(self.kwargs, self.pivot.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.on = 'foo'
        self.on_columns = 'bar', 'baz'
        self.kwargs = {'answer': 42}
        self.pivot = Pivot(self.on, self.on_columns, **self.kwargs)

    def test_callable(self):
        self.assertTrue(callable(self.pivot))

    def test_drop_called(self):
        df = Mock()
        _ = self.pivot(df)
        df.pivot.assert_called_once_with(
            self.on,
            self.on_columns,
            **self.kwargs
        )

    def test_return_value(self):
        df = Mock()
        df.pivot = Mock(return_value='answer')
        actual = self.pivot(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        pivot = Pivot('foo')
        expected = "Pivot('foo', None)"
        self.assertEqual(expected, repr(pivot))

    def test_custom_repr(self):
        pivot = Pivot(pl.col('bar'), ['foo', 'baz'])
        expected = "Pivot(PolarsExpr, ['foo', 'baz'])"
        self.assertEqual(expected, repr(pivot))

    def test_pickle_works(self):
        drop = Pivot(pl.col('bar'))
        _ = pickle.loads(pickle.dumps(drop))


if __name__ == '__main__':
    unittest.main()
