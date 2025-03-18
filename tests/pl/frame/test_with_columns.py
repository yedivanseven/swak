import pickle
import unittest
from unittest.mock import Mock
import polars as pl
from swak.pl import WithColumns


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.with_columns = WithColumns()

    def test_has_exprs(self):
        self.assertTrue(hasattr(self.with_columns, 'exprs'))

    def test_exprs(self):
        self.assertTupleEqual((), self.with_columns.exprs)

    def test_has_named_exprs(self):
        self.assertTrue(hasattr(self.with_columns, 'named_exprs'))

    def test_named_exprs(self):
        self.assertDictEqual({}, self.with_columns.named_exprs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.exprs = 'foo', 'bar'
        self.named_exprs = {'baz': 1, 'answer': 42}
        self.with_columns = WithColumns(*self.exprs, **self.named_exprs)

    def test_exprs(self):
        self.assertTupleEqual(self.exprs, self.with_columns.exprs)

    def test_named_exprs(self):
        self.assertDictEqual(self.named_exprs, self.with_columns.named_exprs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.exprs = 'foo', 'bar'
        self.named_exprs = {'baz': 1}
        self.with_columns = WithColumns(*self.exprs, **self.named_exprs)

    def test_callable(self):
        self.assertTrue(callable(self.with_columns))

    def test_with_columns_called(self):
        df = Mock()
        _ = self.with_columns(df)
        df.with_columns.assert_called_once_with(
            *self.exprs,
            **self.named_exprs
        )

    def test_return_value(self):
        df = Mock()
        df.with_columns = Mock(return_value='result')
        actual = self.with_columns(df)
        self.assertEqual('result', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        with_columns = WithColumns()
        expected = 'WithColumns()'
        self.assertEqual(expected, repr(with_columns))

    def test_custom_repr(self):
        with_columns = WithColumns('foo', 'bar', answer=42)
        expected = "WithColumns('foo', 'bar', answer=42)"
        self.assertEqual(expected, repr(with_columns))

    def test_pickle_works(self):
        with_columns = WithColumns(
            pl.col('col1').mean(),
            col=pl.col('col2').max()
        )
        _ = pickle.loads(pickle.dumps(with_columns))


if __name__ == '__main__':
    unittest.main()
