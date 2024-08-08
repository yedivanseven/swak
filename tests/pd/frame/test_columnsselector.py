import pickle
import unittest
from unittest.mock import MagicMock
import pandas as pd
from swak.pd import ColumnsSelector


class TestAttributeUsage(unittest.TestCase):

    def setUp(self) -> None:
        self.mock = MagicMock()
        self.df = pd.DataFrame([[1, 5], [2, 6], [3, 7], [4, 8]])
        self.empty = pd.DataFrame(columns=[0, 1])

    def test_empty(self):
        select = ColumnsSelector()
        self.assertTrue(hasattr(select, 'cols'))
        self.assertTupleEqual((), select.cols)

    def test_one_int_column(self):
        select = ColumnsSelector(0)
        self.assertTrue(hasattr(select, 'cols'))
        self.assertTupleEqual((0,), select.cols)

    def test_one_str_column(self):
        select = ColumnsSelector('foo')
        self.assertTrue(hasattr(select, 'cols'))
        self.assertTupleEqual(('foo',), select.cols)

    def test_one_int_and_one_str_column(self):
        select = ColumnsSelector(0, 'foo')
        self.assertTrue(hasattr(select, 'cols'))
        self.assertTupleEqual((0, 'foo'), select.cols)

    def test_one_str_and_one_int_column(self):
        select = ColumnsSelector('foo', 0)
        self.assertTrue(hasattr(select, 'cols'))
        self.assertTupleEqual(('foo', 0), select.cols)

    def test_int_tuple_and_str_columns(self):
        select = ColumnsSelector([0, 1], 'foo', 'bar')
        self.assertTrue(hasattr(select, 'cols'))
        self.assertTupleEqual((0, 1, 'foo', 'bar'), select.cols)

    def test_str_tuple_and_int_columns(self):
        select = ColumnsSelector(['foo', 'bar'], 0, 1)
        self.assertTrue(hasattr(select, 'cols'))
        self.assertTupleEqual(('foo', 'bar', 0, 1), select.cols)

    def test_non_hashables_raise(self):
        with self.assertRaises(TypeError):
            _ = ColumnsSelector([[], []])

    def test_non_hashable_args_raise(self):
        with self.assertRaises(TypeError):
            _ = ColumnsSelector(1, [])

    def test_callable_empty(self):
        select = ColumnsSelector()
        self.assertTrue(callable(select))

    def test_callable_one_column(self):
        select = ColumnsSelector(0)
        self.assertTrue(callable(select))

    def test_callable_two_columns(self):
        select = ColumnsSelector(0, 'test')
        self.assertTrue(callable(select))

    def test_getitem_empty(self):
        select = ColumnsSelector()
        _ = select(self.mock)
        self.mock.__getitem__.assert_called_once()
        self.mock.__getitem__.assert_called_once_with([])

    def test_getitem_one_column(self):
        select = ColumnsSelector(0)
        _ = select(self.mock)
        self.mock.__getitem__.assert_called_once()
        self.mock.__getitem__.assert_called_once_with([0])

    def test_getitem_two_columns(self):
        select = ColumnsSelector(0, 'test')
        _ = select(self.mock)
        self.mock.__getitem__.assert_called_once()
        self.mock.__getitem__.assert_called_once_with([0, 'test'])

    def test_return_type_empty(self):
        select = ColumnsSelector()
        selection = select(self.df)
        self.assertIsInstance(selection, pd.DataFrame)

    def test_return_type_one_column(self):
        select = ColumnsSelector(0)
        selection = select(self.df)
        self.assertIsInstance(selection, pd.DataFrame)

    def test_return_type_two_columns(self):
        select = ColumnsSelector(0, 1)
        selection = select(self.df)
        self.assertIsInstance(selection, pd.DataFrame)

    def test_return_value_empty(self):
        select = ColumnsSelector()
        selection = select(self.df)
        pd.testing.assert_frame_equal(self.df[[]], selection)

    def test_return_value_one_column(self):
        select = ColumnsSelector(0)
        selection = select(self.df)
        pd.testing.assert_frame_equal(self.df[[0]], selection)

    def test_return_value_two_columns(self):
        select = ColumnsSelector(0, 1)
        selection = select(self.df)
        pd.testing.assert_frame_equal(self.df[[0, 1]], selection)

    def test_empty_empty(self):
        select = ColumnsSelector()
        selection = select(self.empty)
        self.assertTrue(selection.empty)
        pd.testing.assert_frame_equal(self.empty[[]], selection)

    def test_empty_one_column(self):
        select = ColumnsSelector(0)
        selection = select(self.empty)
        self.assertTrue(selection.empty)
        pd.testing.assert_frame_equal(self.empty[[0]], selection)

    def test_empty_two_columns(self):
        select = ColumnsSelector(0, 1)
        selection = select(self.empty)
        self.assertTrue(selection.empty)
        pd.testing.assert_frame_equal(self.empty, selection)

    def test_empty_empty_empty(self):
        select = ColumnsSelector()
        empty = pd.DataFrame()
        selection = select(empty)
        self.assertTrue(selection.empty)
        pd.testing.assert_frame_equal(empty, selection)


class TestMisc(unittest.TestCase):

    def test_repr(self):
        select = ColumnsSelector(0, 'foo')
        self.assertEqual("ColumnsSelector(0, 'foo')", repr(select))

    def test_pickle_works(self):
        select = ColumnsSelector(0, 'foo')
        _ = pickle.dumps(select)


if __name__ == '__main__':
    unittest.main()
