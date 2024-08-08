import pickle
import unittest
from unittest.mock import MagicMock
import pandas as pd
from swak.pd import ColumnSelector


class TestAttributeUsage(unittest.TestCase):

    def setUp(self) -> None:
        self.mock = MagicMock()
        self.df = pd.DataFrame([[1, 5], [2, 6], [3, 7], [4, 8]])
        self.empty = pd.DataFrame(columns=[0, 1])

    def test_attribute(self):
        select = ColumnSelector(0)
        self.assertTrue(hasattr(select, 'col'))
        self.assertEqual(0, select.col)

    def test_callable(self):
        select = ColumnSelector(0)
        self.assertTrue(callable(select))

    def test_getitem(self):
        select = ColumnSelector(0)
        _ = select(self.mock)
        self.mock.__getitem__.assert_called_once()
        self.mock.__getitem__.assert_called_once_with(0)

    def test_return_type(self):
        select = ColumnSelector(0)
        column = select(self.df)
        self.assertIsInstance(column, pd.Series)

    def test_return_value(self):
        select = ColumnSelector(1)
        column = select(self.df)
        pd.testing.assert_series_equal(self.df[1], column)

    def test_empty(self):
        select = ColumnSelector(1)
        column = select(self.empty)
        pd.testing.assert_series_equal(self.empty[1], column)
        self.assertEqual(0, column.size)

    def test_non_hashables_raise(self):
        with self.assertRaises(TypeError):
            _ = ColumnSelector([])


class TestMisc(unittest.TestCase):

    def test_repr_non_str(self):
        select = ColumnSelector(1)
        self.assertEqual('ColumnSelector(1)', repr(select))

    def test_repr_str(self):
        select = ColumnSelector('1')
        self.assertEqual("ColumnSelector('1')", repr(select))

    def test_pickle_works(self):
        select = ColumnSelector(1)
        _ = pickle.dumps(select)


if __name__ == '__main__':
    unittest.main()
