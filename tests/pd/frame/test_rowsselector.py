import pickle
import unittest
from unittest.mock import MagicMock
import pandas as pd
from swak.pd import RowsSelector


def rows(df):
    return df[0] > 2


def row(df):
    return df[0] > 3


class TestAttributeUsage(unittest.TestCase):

    def setUp(self) -> None:
        self.mock = MagicMock()
        self.df = pd.DataFrame([[1, 5], [2, 6], [3, 7], [4, 8]])
        self.empty = pd.DataFrame(columns=[0, 1])

    def test_attribute(self):
        select = RowsSelector(rows)
        self.assertTrue(hasattr(select, 'condition'))
        self.assertIs(rows, select.condition)

    def test_callable(self):
        select = RowsSelector(rows)
        self.assertTrue(callable(select))

    def test_getitem(self):
        select = RowsSelector(rows)
        _ = select(self.mock)
        self.mock.__getitem__.assert_called_once()
        self.mock.__getitem__.assert_called_once_with(rows)

    def test_return_type(self):
        select = RowsSelector(rows)
        selected = select(self.df)
        self.assertIsInstance(selected, pd.DataFrame)

    def test_return_value(self):
        select = RowsSelector(rows)
        selected = select(self.df)
        pd.testing.assert_frame_equal(self.df.loc[2:, :], selected)

    def test_single_row_return_type(self):
        select = RowsSelector(row)
        selected = select(self.df)
        self.assertIsInstance(selected, pd.DataFrame)

    def test_single_row_return_value(self):
        select = RowsSelector(row)
        selected = select(self.df)
        pd.testing.assert_frame_equal(self.df.loc[3:, :], selected)

    def test_empty(self):
        select = RowsSelector(row)
        selected = select(self.empty)
        self.assertIsInstance(selected, pd.DataFrame)
        self.assertTrue(selected.empty)


class TestMisc(unittest.TestCase):

    def test_repr(self):
        select = RowsSelector(rows)
        self.assertEqual('RowsSelector(rows)', repr(select))

    def test_pickle_works_with_function(self):
        select = RowsSelector(rows)
        _ = pickle.dumps(select)

    def test_pickle_raises_with_lambda(self):
        select = RowsSelector(lambda df: df[0] > 2)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(select)


if __name__ == '__main__':
    unittest.main()
