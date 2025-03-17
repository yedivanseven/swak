import pickle
import unittest
from unittest.mock import Mock
import polars as pl
from swak.pl import Select


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.select = Select()

    def test_has_exprs(self):
        self.assertTrue(hasattr(self.select, 'exprs'))

    def test_exprs(self):
        self.assertTupleEqual((), self.select.exprs)

    def test_has_named_exprs(self):
        self.assertTrue(hasattr(self.select, 'named_exprs'))

    def test_named_exprs(self):
        self.assertDictEqual({}, self.select.named_exprs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.exprs = 'foo', 'bar'
        self.named_exprs = {'baz': 1, 'answer': 42}
        self.select = Select(*self.exprs, **self.named_exprs)

    def test_exprs(self):
        self.assertTupleEqual(self.exprs, self.select.exprs)

    def test_named_exprs(self):
        self.assertDictEqual(self.named_exprs, self.select.named_exprs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.exprs = 'foo', 'bar'
        self.named_exprs = {'baz': 1}
        self.select = Select(*self.exprs, **self.named_exprs)

    def test_callable(self):
        self.assertTrue(callable(self.select))

    def test_select_called(self):
        df = Mock()
        _ = self.select(df)
        df.select.assert_called_once_with(*self.exprs, **self.named_exprs)

    def test_return_value(self):
        df = Mock()
        df.select = Mock(return_value='answer')
        actual = self.select(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        select = Select()
        expected = 'Select()'
        self.assertEqual(expected, repr(select))

    def test_custom_repr(self):
        select = Select('foo', 'bar', answer=42)
        expected = "Select('foo', 'bar', answer=42)"
        self.assertEqual(expected, repr(select))

    def test_pickle_works(self):
        select = Select(pl.col('col1').mean(), col=pl.col('col2').max())
        _ = pickle.loads(pickle.dumps(select))


if __name__ == '__main__':
    unittest.main()
